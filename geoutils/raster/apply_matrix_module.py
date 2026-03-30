from __future__ import annotations

from typing import (
    Any,
    Callable,
    Literal,
    overload,
)

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio


import geoutils as gu
from geoutils.multiproc.chunked import _chunks2d_from_chunksizes_shape
from geoutils.multiproc.mparray import MultiprocConfig, _write_multiproc_result
from geoutils.raster.apply_matrix_function import (
    _check_nodata_dtype,
    _reproject_horizontal_shift_samecrs,
    _apply_matrix_rst,
    _build_geotiling_and_meta_apply_matrix,
)

from geoutils.raster.transformation import _rio_reproject

from geoutils._misc import import_optional, silence_rasterio_message


# Dask as optional dependency
try:
    import dask.array as da
    from dask import delayed
except ImportError:

    da = None

    def delayed(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake delayed decorator if dask is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


@overload
def apply_matrix(
    elev: NDArrayf,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resample: bool = True,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    transform: rio.transform.Affine = None,
    z_name: str = "z",
    multiproc_config: gu.raster.MultiprocConfig | None = None,
    **kwargs: Any,
) -> tuple[NDArrayf, affine.Affine]: ...


@overload
def apply_matrix(
    elev: gu.Raster | gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resample: bool = True,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    transform: rio.transform.Affine = None,
    z_name: str = "z",
    multiproc_config: gu.raster.MultiprocConfig | None = None,
    **kwargs: Any,
) -> gu.Raster | gpd.GeoDataFrame: ...


def apply_matrix(
    elev: gu.Raster | NDArrayf | gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resample: bool = True,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    src_transform: rio.transform.Affine = None,
    dst_transform: rio.transform.Affine = None,
    z_name: str = "z",
    multiproc_config: gu.raster.MultiprocConfig | None = None,
    **kwargs: Any,
) -> tuple[NDArrayf, affine.Affine] | gu.Raster | gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud or 2.5D DEM.

    For an elevation point cloud, the transformation is exact.

    For a DEM, it requires re-gridding because the affine-transformed point cloud of the DEM does not fall onto a
    regular grid anymore (except if the affine transformation only has translations). For this, this function uses the
    three following methods:

    1. For transformations with only translations, the transform is updated and vertical shift added to the array,

    2. For transformations with a small rotation (20 degrees or less for all axes), this function maps which 2D
    point coordinates will fall back exactly onto the original DEM grid coordinates after affine transformation by
    searching iteratively using the invert affine transformation and 2D point regular-grid interpolation on the
    original DEM (see geoutils.Raster.interp_points, or scipy.interpolate.interpn),

    3. For transformations with large rotations (20 degrees or more), scipy.interpolate.griddata is used to
    re-grid the irregular affine-transformed 3D point cloud using Delauney triangulation interpolation (slower).

    :param elev: Elevation point cloud or DEM to transform, either a 2D array (requires transform) or
        geodataframe (requires z_name).
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param resample: (For translations) If set to True, will resample output on the translated grid to match the input
        transform. Otherwise, only the transform will be updated and no resampling is done.
    :param resampling: Point interpolation method, one of 'nearest', 'linear', 'cubic', or 'quintic'. For more
        information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn. Default is linear.
    :param transform: Geotransform of the DEM, only for DEM passed as 2D array.
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
    :param kwargs: Keywords passed to _apply_matrix_rst for testing.

    :return: Affine transformed elevation point cloud or DEM.
    """

    mp_backend = multiproc_config is not None
    # The check below can only run on Xarray
    dask_backend = (
        da is not None and isinstance(elev, gu.raster.xr_accessor.RasterAccessor) and elev._chunks is not None
    )

    if multiproc_config and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from reproject(). To use Multiprocessing, open the file without chunks."
        )
    # Apply matrix to elevation point cloud
    if isinstance(elev, gpd.GeoDataFrame):
        return _apply_matrix_pts(epc=elev, matrix=matrix, invert=invert, centroid=centroid, z_name=z_name)
    # Or apply matrix to raster (often requires re-gridding)
    else:

        if isinstance(elev, gu.Raster):
            src_transform = elev.transform
            dem = elev.data.filled(np.nan)
        elif isinstance(elev, gu.raster.xr_accessor.RasterAccessor):
            src_transform = elev.transform
            # dem = elev.data.filled(np.nan)
        else:
            dem = elev

        # If using Multiprocessing backend, process and return None (files written on disk)
        if mp_backend or dask_backend:

            # 2/ Check user input for nodata and dtype
            dtype, src_nodata, nodata = _check_nodata_dtype(
                source_raster=elev,
                nodata=elev.nodata,
                dtype=elev.dtype,
                force_source_nodata=None,
            )

            # 3/ Store georeferencing parameters for apply_matrix
            apply_matrix_kwargs = {
                "matrix": matrix,
                "invert": invert,
                "centroid": centroid,
                "resample": resample,
                "resampling": resampling,
                "src_transform": src_transform,
                "src_nodata": src_nodata,
                "src_crs": elev.crs,
            }

            if multiproc_config:
                _multiproc_apply_matrix(elev, mp_config=multiproc_config, **apply_matrix_kwargs)
                new_raster = gu.Raster(multiproc_config.outfile)
                new_raster.set_mask(new_raster == src_nodata)
                print (new_raster)
                return new_raster

            elif da is not None and isinstance(elev.data, da.Array):
                dst_arr = _dask_apply_matrix(darr=elev.data, **apply_matrix_kwargs)

                if dst_transform is None:
                    if resample == True:
                        dst_transform = src_transform


                return gu.raster.xr_accessor.RasterAccessor.from_array(
                    data=dst_arr, transform=dst_transform, crs=elev.crs, nodata=src_nodata, area_or_point=elev.area_or_point,
                    tags=elev.tags
                )
        else:

            # Then, if resample is True, we reproject the DEM from its out_transform onto the transform

            if dst_transform is None:
                if resample == True:
                    dst_transform = src_transform

            applied_dem, out_transform = _apply_matrix_rst(
                dem=dem,
                src_transform=src_transform,
                matrix=matrix,
                invert=invert,
                centroid=centroid,
                resampling=resampling,
                resample=resample,
                out_transform=dst_transform,
                **kwargs,
            )



            # We return a raster if input was a raster
            if isinstance(elev, gu.Raster):
                applied_dem = gu.Raster.from_array(applied_dem, out_transform, elev.crs, elev.nodata)
                return applied_dem
            return applied_dem, out_transform


def _apply_matrix_per_block(
    *src_arrs: tuple[NDArrayNum],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    src_nodata,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """

    is_multiband = combined_meta["dst_count"] >= 2

    # If no source chunk intersects, we return a chunk of destination nodata values
    if len(src_arrs) == 0:
        # We can use float32 to return NaN, will be cast to other floating type later if that's not source array dtype
        dst_shape = (
            (combined_meta["dst_count"], *combined_meta["dst_shape"]) if is_multiband else combined_meta["dst_shape"]
        )
        dst_arr = np.zeros(dst_shape, dtype=np.dtype("float32"))
        dst_arr[:] = np.nan
        return dst_arr

    # First, we build an empty array with the combined shape, only with nodata values
    shape = (src_arrs[0].shape[0], *combined_meta["src_shape"]) if is_multiband else combined_meta["src_shape"]

    comb_src_arr = np.full(shape, src_nodata, dtype=src_arrs[0].dtype)
    if np.ma.isMaskedArray(src_arrs[0]):
        comb_src_arr = np.ma.masked_array(data=comb_src_arr)

    # Then fill it with the source chunks values
    for arr, bid in zip(src_arrs, block_ids):
        comb_src_arr[..., bid["rys"] : bid["rye"], bid["rxs"] : bid["rxe"]] = arr


    # Now, we can simply call Rasterio!
    # We build the combined transform from tuple
    src_transform = rio.transform.Affine(*combined_meta["src_transform"])
    dst_transform = rio.transform.Affine(*combined_meta["dst_transform"])
    # Apply matrix wrapper

    kwargs["src_transform"] = src_transform
    kwargs["dst_transform"] = dst_transform

    dst_arr, out_transform = apply_matrix(elev=comb_src_arr, **kwargs)  # type: ignore
    dst_arr = dst_arr[:combined_meta["dst_shape"][0], :combined_meta["dst_shape"][1]]


    return dst_arr


def _wrapper_multiproc_nb_valids_per_block(rst: Raster, tile_idx: NDArrayNum) -> int:
    """Count valid values in one tile out-of-memory."""
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))
    arr = rst_block.data

    if np.issubdtype(arr.dtype, np.bool_):
        return int(np.count_nonzero(arr))
    return int(np.count_nonzero(~get_mask_from_array(arr)))


def _wrapper_multiproc_apply_matrix_per_block(
    rst: Raster,
    src_block_ids: list[dict[str, int]],
    dst_block_id: dict[str, int],
    idx_d2s: list[int],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Wrapper to use Delayed reprojection per destination block
    (also rebuilds a square array combined from intersecting source blocks)."""

    # Get source array block for each destination block
    s = src_block_ids
    src_arrs = (rst.icrop(bbox=(s[idx]["xs"], s[idx]["ys"], s[idx]["xe"], s[idx]["ye"])).data for idx in idx_d2s)

    # Call reproject per block
    dst_block_arr = _apply_matrix_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)
    return dst_block_arr, (dst_block_id["ys"], dst_block_id["ye"], dst_block_id["xs"], dst_block_id["xe"])


def _multiproc_apply_matrix(
    rst: RasterType,
    mp_config: MultiprocConfig,
    src_crs,
    **kwargs: Any,
) -> tuple[NDArrayf, rio.transform.Affine]:

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_chunks = _chunks2d_from_chunksizes_shape(
        chunksizes=(mp_config.chunk_size, mp_config.chunk_size), shape=rst.shape
    )

    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta_apply_matrix(
            src_count=rst.count,
            src_shape=rst.shape,
            src_transform=kwargs["src_transform"],
            src_crs=src_crs,
            dst_shape=rst.shape,
            dst_transform=kwargs["src_transform"],
            dst_crs=src_crs,
            src_chunks=src_chunks,
            dst_chunksizes=(mp_config.chunk_size, mp_config.chunk_size),
            matrix=kwargs["matrix"]
        )
    )

    # Get location of destination blocks to write file
    dst_block_ids = np.array(dst_geotiling.get_block_locations())

    # Create tasks for multiprocessing
    tasks = []

    for i in range(len(dest2source)):
        tasks.append(
            mp_config.cluster.launch_task(
                fun=_wrapper_multiproc_apply_matrix_per_block,
                args=[
                    rst,
                    src_block_ids,
                    dst_block_ids[i],
                    dest2source[i],
                    meta_params[i][1],
                    meta_params[i][0],
                ],
                kwargs=kwargs,
            )
        )

    # Retrieve metadata for saving file
    file_metadata = {
        "width": rst.shape[1],
        "height": rst.shape[0],
        "count": rst.count,
        "crs": rst.crs,
        "transform": rst.transform,
        "dtype": rst.dtype,
        "nodata": rst.nodata,
    }

    # Create a new raster file to save the processed results
    _write_multiproc_result(tasks, mp_config, file_metadata)


@delayed
def _delayed_apply_matrix_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Delayed reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """
    return _apply_matrix_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)


def _dask_apply_matrix(
    darr: da.Array,
    src_crs,
    **kwargs: Any,
) -> da.Array:

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Define the chunking
    # For source, we can use the .chunks attribute
    src_chunks = darr.chunks[-2:]  # In case input is multi-band

    dst_chunksizes = (darr.chunksize[-2], darr.chunksize[-1])  # In case input is multi-band

    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta_apply_matrix(
            src_count=darr.shape[0] if darr.ndim == 3 else 1,
            src_shape=darr.shape[-2:],  # In case input is multi-band
            src_transform=kwargs["src_transform"],
            src_crs=src_crs,
            dst_shape=darr.shape[-2:],  # In case input is multi-band
            dst_transform=kwargs["src_transform"],
            dst_crs=src_crs,
            src_chunks=src_chunks,
            dst_chunksizes=dst_chunksizes,
            matrix=kwargs["matrix"]
        )
    )

    dst_block_ids = np.array(dst_geotiling.get_block_locations())

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks_delayed = darr.to_delayed()

    # Spatial block grid shape (from spatial chunks)
    is_multiband = darr.ndim == 3
    ny_src = len(src_chunks[0])
    nx_src = len(src_chunks[1])
    src_yi, src_xi = np.unravel_index(np.arange(ny_src * nx_src), shape=(ny_src, nx_src))
    # Normalize band groups:
    # - 2D: one pseudo group (bb=None, nb=0)
    # - 3D: real band blocks with their sizes
    band_groups: list[tuple[int | None, int]] = (
        [(None, 0)] if not is_multiband else [(bb, int(sz)) for bb, sz in enumerate(darr.chunks[0])]
    )
    # Output data type
    out_dtype = np.dtype(kwargs.get("dtype", darr.dtype))

    # Helper function to support both 2D and 3D cases
    def _dst_block_as_da(i: int) -> da.Array:
        """Build destination block as a Dask array (2D or 3D)."""
        shp2 = dst_block_geogrids[i].shape  # (ydst, xdst)

        # Spatial source coords for this destination tile
        coords = [(src_yi[j], src_xi[j]) for j in dest2source[i]]

        def _src_chunks_for_group(bb: int | None) -> list[Any]:
            # Accounting for the fact that blocks_delayed is either (ny,nx) or (nb,ny,nx)
            if bb is None:
                return [blocks_delayed[y, x] for (y, x) in coords]
            return [blocks_delayed[bb, y, x] for (y, x) in coords]

        def _one_group(bb: int | None, nb: int) -> da.Array:
            r = _delayed_apply_matrix_per_block(
                *_src_chunks_for_group(bb),
                block_ids=meta_params[i][1],
                combined_meta=meta_params[i][0],
                **kwargs,
            )
            shape = shp2 if bb is None else (nb, *shp2)

            # We define the expected output shape and dtype to simplify things for Dask
            return da.from_delayed(r, shape=shape, dtype=out_dtype)

        # Build per-group outputs then concatenate along band axis if needed
        groups = [_one_group(bb, nb) for (bb, nb) in band_groups]
        return groups[0] if len(groups) == 1 else da.concatenate(groups, axis=0)

    # Run the delayed reprojection, looping for each destination block-band (2D block and 1D band-chunk)
    list_reproj_da = [_dst_block_as_da(i) for i in range(len(dest2source))]
    # Array comes out as flat blocks x chunksize0 (varying) x chunksize1 (varying), so we can't reshape directly
    # We need to unravel the flattened blocks indices to align X/Y, then concatenate all columns, then rows
    ny_dst, nx_dst = len(dst_chunks[0]), len(dst_chunks[1])
    iy, ix = np.unravel_index(np.arange(len(dest2source)), shape=(ny_dst, nx_dst))
    ax_x = 1 if darr.ndim == 2 else 2  # Adjust axes depending on if raster is single-band or multi-band
    ax_y = 0 if darr.ndim == 2 else 1
    rows = [
        da.concatenate([list_reproj_da[k] for k in range(len(list_reproj_da)) if iy[k] == r], axis=ax_x)
        for r in range(ny_dst)
    ]
    concat_all = da.concatenate(rows, axis=ax_y)
    return concat_all
