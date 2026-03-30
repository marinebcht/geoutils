# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functionalities for transformations of raster objects.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

import affine
import logging
import numpy as np
import rasterio as rio
from packaging.version import Version
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
import scipy

from geoutils import profiler
import geoutils
from geoutils._dispatch import _check_match_bbox, _check_match_grid
from geoutils._misc import import_optional, silence_rasterio_message
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum
from geoutils.multiproc.chunked import (
    ChunkedGeoGrid,
    GeoGrid,
    _chunks2d_from_chunksizes_shape,
)
from geoutils.raster.transformation import _combined_blocks_shape_transform
from geoutils.raster.transformation import _translate


if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike, RasterType
    from geoutils.raster.raster import Raster
    from geoutils.vector.vector import VectorLike

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


def _build_geotiling_and_meta_apply_matrix(
    src_count: int,
    src_shape: tuple[int, int],
    src_transform: rio.transform.Affine,
    src_crs: CRS,
    dst_shape: tuple[int, int],
    dst_transform: rio.transform.Affine,
    dst_crs: CRS,
    src_chunks: tuple[tuple[int, ...], tuple[int, ...]],
    dst_chunksizes: tuple[int, int],
    matrix
) -> tuple[
    ChunkedGeoGrid,
    ChunkedGeoGrid,
    tuple[tuple[int, ...], tuple[int, ...]],
    list[list[int]],
    list[dict[str, int]],
    list[tuple[dict[str, Any], list[dict[str, int]]]],
    list[GeoGrid],
]:
    """
    Constructs georeferenced tiling information and reprojection metadata for both source and destination grids,
    used to support block-wise reprojection operations (e.g. with multiprocessing or dask).

    This function performs the following:

    1. Constructs `GeoGrid` and `ChunkedGeoGrid` objects for source and destination rasters,
       based on provided shape, transform, CRS, and chunk sizes.
    2. Computes spatial footprints for each chunk in both grids, and determines which
       source chunks intersect each destination chunk (with a buffer to ensure overlap).
    3. For each destination chunk, calculates metadata required for reprojection, including:
       - The combined shape and transform of all intersecting source chunks.
       - The specific shape and transform of the destination block.

    :return: A tuple containing:
        - Source `ChunkedGeoGrid`
        - Destination `ChunkedGeoGrid`
        - Destination chunks
        - Mapping from destination to intersecting source block indices
        - Array of source block locations
        - List of metadata dictionaries per destination block
        - List of destination `GeoGrid` blocks
    """
    # 1/ Define source and destination chunked georeferenced grid through simple classes storing CRS/transform/shape,
    # which allow to consistently derive shape/transform for each block and their CRS-projected footprints

    # Define GeoGrids for source/destination array
    src_geogrid = GeoGrid(transform=src_transform, shape=src_shape, crs=src_crs)
    dst_geogrid = GeoGrid(transform=dst_transform, shape=dst_shape, crs=dst_crs)

    # Create tilings
    src_geotiling = ChunkedGeoGrid(grid=src_geogrid, chunks=src_chunks)
    dst_chunks = _chunks2d_from_chunksizes_shape(chunksizes=dst_chunksizes, shape=dst_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)

    # 2/ Get bounds of tiles in CRS of destination array, with a buffer of 2 pixels for destination ones to ensure
    # overlap, then map indexes of source blocks that intersect a given destination block

    src_boxes = [box(*gg.bounds_projected(crs=dst_crs)) for gg in src_geotiling.get_blocks_as_geogrids()]

    def _wrapper_multiproc_nb_valids_per_block(rst: Raster, tile_idx: NDArrayNum) -> int:
        """Count valid values in one tile out-of-memory."""
        rst_block = rst.icrop((tile_idx["xs"], tile_idx["ys"], tile_idx["xe"], tile_idx["ye"]))
        arr = rst_block.data
        return arr.min(), arr.max()

    src_block_ids = src_geotiling.get_block_locations()

    dst_boxes = []
    for k, gg in enumerate(dst_geotiling.get_blocks_as_geogrids()):
        poly = box(*gg.bounds_projected(crs=dst_crs)).buffer(2 * max(dst_geogrid.res))
        xx, yy = poly.exterior.coords.xy
        # zz_min, zz_max = mp_config.cluster.launch_task(fun=_wrapper_multiproc_nb_valids_per_block, args=[rst, src_block_ids[k]], kwargs={})
        # zz = np.ones(len(xx)) * (zz_max - zz_min)
        zz = np.zeros(len(xx))
        dem = _apply_matrix_pts_arr(x=list(xx), y=list(yy), z=list(zz), invert=True, matrix=matrix)
        poly_res = Polygon(zip(dem[0], dem[1]))

        dst_boxes.append(poly_res)

    # Faster to use spatial index over source boxes
    tree = STRtree(src_boxes)

    # For Shapely 2.0: STRtree.query(..., predicate="intersects") is fastest, for earlier versions we filter manually

    # Quick feature check
    try:
        _ = tree.query(dst_boxes[0], predicate="intersects") if dst_boxes else []
        has_predicate = True
    except TypeError:
        has_predicate = False

    # Build mapping: for each destination box, list intersecting source indices
    dest2source: list[list[int]] = []
    if has_predicate:
        # Shapely 2: Query returns indices directly (int array)
        for dst in dst_boxes:
            idx = tree.query(dst, predicate="intersects")
            dest2source.append([int(i) for i in np.asarray(idx).ravel()])
    else:
        # Shapely 1.8: Query returns geometries, so we convert to indices via id() map + filter intersects
        id_to_idx = {id(g): i for i, g in enumerate(src_boxes)}
        for dst in dst_boxes:
            cand_geoms = tree.query(dst)
            matches = [id_to_idx[id(g)] for g in cand_geoms if dst.intersects(g)]
            dest2source.append(matches)


    # 3/ To reconstruct a square source array during chunked reprojection, we need to derive the combined shape and
    # transform of each tuples of source blocks
    src_block_ids = src_geotiling.get_block_locations()
    meta_params = [
        (
            _combined_blocks_shape_transform(sub_block_ids=[src_block_ids[i] for i in sbid], src_geogrid=src_geogrid)
            if len(sbid) > 0
            else ({}, [])
        )
        for sbid in dest2source
    ]

    # Append dst shape/transform to metadata
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()
    for i, (c, _) in enumerate(meta_params):
        c.update(
            {
                "dst_shape": dst_block_geogrids[i].shape,
                "dst_transform": tuple(dst_block_geogrids[i].transform),
                "dst_count": src_count,
            }
        )

    return src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids


def _reproject_horizontal_shift_samecrs(
    raster_arr: NDArrayf,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine = None,
    return_interpolator: bool = False,
    resampling: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> NDArrayf | Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf]:
    """
    Reproject a raster only for a horizontal shift (transform update) in the same CRS.

    This function exists independently of Raster.reproject() because Rasterio has unexplained reprojection issues
    that can create non-negligible sub-pixel shifts that should be crucially avoided for coregistration.
    See https://github.com/rasterio/rasterio/issues/2052#issuecomment-2078732477.

    Here we use SciPy interpolation instead, modified for nodata propagation in geoutils.interp_points().
    """

    # We are reprojecting the raster array relative to itself without changing its pixel interpretation, so we can
    # force any pixel interpretation (area_or_point) without it having any influence on the result, here "Area"
    from geoutils.raster.referencing import _coords

    if not return_interpolator:
        coords_dst = _coords(transform=dst_transform, area_or_point="Area", shape=raster_arr.shape)
        # Flatten the arrays (only 1D supported in rowcol/xy after Rasterio 1.4)
        coords_dst = (coords_dst[0].ravel(), coords_dst[1].ravel())
    # If we just want the interpolator, we don't need to coordinates of destination points
    else:
        coords_dst = None

    from geoutils.interface.interpolation import _interp_points_base

    output = _interp_points_base(
        array=raster_arr,
        area_or_point="Area",
        transform=src_transform,
        points=coords_dst,
        method=resampling,
        return_interpolator=return_interpolator,
    )

    # Reshape output
    if coords_dst is not None:
        output = output.reshape(np.shape(raster_arr))

    return output


def _check_nodata_dtype(
    source_raster: RasterType,
    nodata: int | float | None,
    dtype: DTypeLike | None,
    force_source_nodata: int | float | None,
) -> tuple[DTypeLike, int | float | None, int | float | None]:
    """Check user inputs of reproject regarding nodata and data type."""

    # Set output dtype
    if dtype is None:
        # Warning: this will not work for multiple bands with different dtypes
        dtype = source_raster.dtype

    # --- Set source nodata if provided -- #
    if force_source_nodata is None:
        src_nodata = source_raster.nodata
    else:
        src_nodata = force_source_nodata
        # Raise warning if a different nodata value exists for this raster than the forced one (not None)
        if source_raster.nodata is not None:
            warnings.warn(
                "Forcing source nodata value of {} despite an existing nodata value of {} in the raster. "
                "To silence this warning, use self.set_nodata() before reprojection instead of forcing.".format(
                    force_source_nodata, source_raster.nodata
                )
            )

    # --- Set destination nodata if provided -- #
    # This is needed in areas not covered by the input data.
    # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
    if nodata is None:
        nodata = source_raster.nodata
        if nodata is None:
            nodata = _default_nodata(dtype)
            # If nodata is already being used, raise a warning.
            if not source_raster.is_loaded:
                warnings.warn(
                    f"For reprojection, nodata must be set. Setting default nodata to {nodata}. You may "
                    f"set a different nodata with `nodata`."
                )

            elif nodata in source_raster.data:
                warnings.warn(
                    f"For reprojection, nodata must be set. Default chosen value {nodata} exists in "
                    f"self.data. This may have unexpected consequences. Consider setting a different nodata with "
                    f"self.set_nodata()."
                )

    return dtype, src_nodata, nodata


def translations_rotations_from_matrix(
    matrix: NDArrayf, return_degrees: bool = True
) -> tuple[float, float, float, float, float, float]:
    """
    Extract 3 translations (unit of coordinates) and 3 rotations (degrees or radians) from rigid affine matrix.

    The extracted euler rotations use the extrinsic convention.

    :param matrix: Rigid affine matrix of transformation.
    :param return_degrees: Whether to return rotations in degrees, otherwise radians.

    :return: Translations in the X, Y and Z direction and rotations around the X, Y and Z directions.
    """

    # Extract translations
    t1, t2, t3 = matrix[:3, 3]

    # Get rotations from affine matrix
    rots = _matrix_to_euler(matrix[:3, :3])
    if return_degrees:
        rots = np.rad2deg(np.array(rots))

    # Extract rotations
    alpha1, alpha2, alpha3 = rots

    return t1, t2, t3, alpha1, alpha2, alpha3


def _matrix_to_euler(rotation_matrix: NDArrayf, atol: float = 10e-8) -> tuple[float, float, float]:
    """
    Affine matrix to extrinsic Euler angles.

    :param rotation_matrix: Rotation matrix.

    :return: Euler extrinsic angles in radians (rotations about X, Y and Z).
    """

    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=atol):
        raise ValueError("Matrix is not orthogonal")

    if abs(rotation_matrix[2, 0]) < 1 - atol:
        beta = -np.arcsin(rotation_matrix[2, 0])
        cb = np.cos(beta)

        alpha = np.arctan2(rotation_matrix[2, 1] / cb, rotation_matrix[2, 2] / cb)
        gamma = np.arctan2(rotation_matrix[1, 0] / cb, rotation_matrix[0, 0] / cb)

    # Gimbal lock
    else:
        beta = np.pi / 2 if rotation_matrix[2, 0] <= -1 else -np.pi / 2
        alpha = 0.0
        gamma = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

    return float(alpha), float(beta), float(gamma)


def _iterate_affine_regrid_small_rotations(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    centroid: tuple[float, float, float] | None = None,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
) -> tuple[NDArrayf, rio.transform.Affine]:
    """
    Iterative process to find the best reprojection of affine transformation for small rotations.

    Faster than regridding point cloud by triangulation of points (for instance with scipy.interpolate.griddata).
    """

    # Convert DEM to elevation point cloud, keeping all exact grid coordinates X/Y even for NaNs
    dem_rst = geoutils.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
    epc = dem_rst.to_pointcloud(data_column_name="z", skip_nodata=False).ds

    # Exact affine transform of elevation point cloud (which yields irregular coordinates in 2D)
    tz0 = _apply_matrix_pts_arr(
        x=epc.geometry.x.values, y=epc.geometry.y.values, z=epc.z.values, matrix=matrix, centroid=centroid
    )[2]

    # We need to find the elevation Z of a transformed DEM at the exact grid coordinates X,Y
    # Which means we need to find coordinates X',Y',Z' of the original DEM that, after the exact affine transform,
    # fall exactly on regular X,Y coordinates

    # 1/ The elevation of the original DEM, Z', is simply a 2D interpolator function of X',Y' (bilinear, typically)
    # (We create the interpolator only once here for computational speed, instead of using Raster.interp_points)
    xycoords = dem_rst.coords(grid=False)
    z_interp = scipy.interpolate.RegularGridInterpolator(
        points=(np.flip(xycoords[1], axis=0), xycoords[0]), values=dem, method=resampling, bounds_error=False
    )

    # 2/ As a first guess of a transformed DEM elevation Z near the grid coordinates, we initialize with the elevations
    # of the nearest point from the transformed elevation point cloud

    # OLD METHOD
    # (Longest step computationally)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
    #     nearest = gpd.sjoin_nearest(epc, trans_epc)
    #
    # # In case several points are found at exactly the same distance, take the mean of their elevations
    # new_z = nearest.groupby(by=nearest.index)["z_left"].mean().values

    # NEW METHOD: Use the transformed elevation instead of searching for a nearest neighbour,
    # is close enough for small rotations! (and only creates a couple more iterations instead of a full search)
    new_z = tz0

    # 3/ We then iterate between two steps until convergence:
    # a/ Use the Z guess to derive invert affine transform X',Y' coordinates for the original DEM,
    # b/ Interpolate Z' at new coordinates X',Y' on the original DEM, and apply affine transform to get updated Z guess

    # Start with full array of X/Y regular coordinates (subset during iterations to improve computational efficiency)
    x = epc.geometry.x.values
    y = epc.geometry.y.values

    # Initialize output z array, and array to store points that have converged
    zfinal = np.ones(len(x), dtype=dem.dtype)
    ind_converged = np.zeros(len(x), dtype=bool)

    # For small rotations, and large DEMs (elevation range smaller than the DEM extent), this converges fast
    max_niter = 20  # Maximum iteration number
    niter_check = 5  # Number of iterations between residual checks
    tolerance = 10 ** (-4)  # Tolerance in X/Y relative to resolution of X/Y
    res_x = dem_rst.res[0]  # Resolution in X
    res_y = dem_rst.res[1]  # Resolution in Y
    niter = 1  # Starting iteration

    while niter < max_niter:

        # Invert X,Y (exact grid coordinates) with Z guess to find X',Y' coordinates on original DEM
        tx, ty = _apply_matrix_pts_arr(x=x, y=y, z=new_z, matrix=matrix, invert=True, centroid=centroid)[:2]

        # Interpolate original DEM at X', Y' to get Z', and convert to point cloud
        tz = z_interp((ty, tx))

        # Transform to see if we fall back on our feet (on the regular grid), or if we need to iterate more
        x0, y0, z0 = _apply_matrix_pts_arr(x=tx, y=ty, z=tz, matrix=matrix, centroid=centroid)

        # Only check residuals after first iteration (to remove NaNs) then every 5 iterations to reduce computing time
        if niter == 1 or niter == niter_check:

            # Compute difference between exact grid coordinates and current coordinates, and stop if tolerance reached
            diff_x = x0 - x
            diff_y = y0 - y

            logging.debug(
                "Residual check at iteration number %d:" "\n    Mean diff x: %f" "\n    Mean diff y: %f",
                niter,
                np.nanmean(np.abs(diff_x)),
                np.nanmean(np.abs(diff_y)),
            )

            # Get index of points below tolerance in both X/Y for this subsample (all points before convergence update)
            # Nodata values are considered having converged
            subind_diff_x = np.logical_or(np.abs(diff_x) < (tolerance * res_x), ~np.isfinite(diff_x))
            subind_diff_y = np.logical_or(np.abs(diff_y) < (tolerance * res_y), ~np.isfinite(diff_y))
            subind_converged = np.logical_and(subind_diff_x, subind_diff_y)

            logging.debug(
                "    Points not within tolerance: %d for X; %d for Y",
                np.count_nonzero(~subind_diff_x),
                np.count_nonzero(~subind_diff_y),
            )

            # If all points left are below convergence, update Z one final time and stop here
            if all(subind_converged):
                zfinal[~ind_converged] = z0
                break
            # Otherwise, save Z for new converged points and keep only not converged in next iterations (for speed)
            else:
                zfinal[~ind_converged] = z0
                x = x[~subind_converged]
                y = y[~subind_converged]
                z0 = z0[~subind_converged]

            # Otherwise, for this check, update convergence status for points not having converged yet
            ind_converged[~ind_converged] = subind_converged

        # If another iteration is required, update Z guess and increment
        new_z = z0
        niter += 1

    # 4/ Write the regular-grid point cloud back into a raster
    epc.z = zfinal  # We just replace the Z of the original grid to ensure exact coordinates
    transformed_dem = dem_rst.from_pointcloud_regular(
        epc, transform=transform, shape=dem.shape, data_column_name="z", nodata=-99999
    )

    return transformed_dem.data.filled(np.nan), transform


def _apply_matrix_rst(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    force_regrid_method: Literal["iterative", "griddata"] | None = None,
    resample = True,
    out_transform = None,
) -> tuple[NDArrayf, rio.transform.Affine]:
    """
    Apply a 3D affine transformation matrix to a 2.5D DEM.

    See details in description of apply_matrix().

    :param dem: DEM to transform.
    :param transform: Geotransform of the DEM.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param resampling: Point interpolation method, one of 'nearest', 'linear', 'cubic', or 'quintic'. For more
    information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn. Default is linear.
    :param force_regrid_method: Force re-gridding method to convert 3D point cloud to 2.5 DEM, only for testing.

    :returns: Transformed DEM, Transform.
    """

    # Invert matrix if required
    if invert:
        matrix = invert_matrix(matrix)

    # Check DEM has valid values
    if np.count_nonzero(np.isfinite(dem)) == 0:
        raise ValueError("Input DEM has all nans.")

    shift_z_only_matrix = np.diag(np.ones(4, float))
    shift_z_only_matrix[2, 3] = matrix[2, 3]

    shift_only_matrix = np.diag(np.ones(4, float))
    shift_only_matrix[:3, 3] = matrix[:3, 3]

    if (np.array_equal(shift_z_only_matrix, matrix) and force_regrid_method is None) or \
        (np.array_equal(shift_only_matrix, matrix) and force_regrid_method is None) :
        # 1/ Check if the matrix only contains a Z correction, in that case only shift the DEM values by the vertical shift
        if np.array_equal(shift_z_only_matrix, matrix) and force_regrid_method is None:
            dem, transform = dem + matrix[2, 3], transform

        # 2/ Check if the matrix contains only translations, in that case only shift the DEM only by translation
        if np.array_equal(shift_only_matrix, matrix) and force_regrid_method is None:
            new_transform = _translate(transform, xoff=matrix[0, 3], yoff=matrix[1, 3])
            dem, transform = dem + matrix[2, 3], new_transform

        # Then, if resample is True, we reproject the DEM from its out_transform onto the transform
        if resample:
            dem = _reproject_horizontal_shift_samecrs(
                dem, src_transform=transform, dst_transform=out_transform, resampling=resampling
            )
            transform = out_transform
        return dem, transform

    # 3/ If matrix contains only small rotations (less than 20 degrees), use the fast iterative reprojection
    rotations = translations_rotations_from_matrix(matrix)[3:]
    if all(np.abs(rot) < 20 for rot in rotations) and force_regrid_method is None or force_regrid_method == "iterative":
        new_dem, transform = _iterate_affine_regrid_small_rotations(
            dem=dem, transform=transform, matrix=matrix, centroid=centroid, resampling=resampling
        )
    else :
        # 4/ Otherwise, use a delauney triangulation interpolation of the transformed point cloud
        # Convert DEM to elevation point cloud, keeping all exact grid coordinates X/Y even for NaNs
        dem_rst = geoutils.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
        epc = dem_rst.to_pointcloud(data_column_name="z").ds
        trans_epc = _apply_matrix_pts(epc, matrix=matrix, centroid=centroid)

        from geoutils.interface.gridding import _grid_pointcloud

        new_dem = _grid_pointcloud(
            trans_epc, grid_coords=dem_rst.coords(grid=False), data_column_name="z", resampling=resampling
        )[0]

    # Then, if resample is True, we reproject the DEM from its out_transform onto the transform
    if resample:
        new_dem = _reproject_horizontal_shift_samecrs(
            new_dem, src_transform=transform, dst_transform=out_transform, resampling=resampling
        )
        transform = out_transform
    return new_dem, transform


def invert_matrix(matrix: NDArrayf, atol: float = 10e-8) -> NDArrayf:
    """
    Invert a transformation matrix.

    :param matrix: Affine transformation matrix.

    :return: Inverted transformation matrix.
    """

    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("Not affine")

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError("Not a rigid transform")

    # Make valid before inversion
    valid_matrix = _make_matrix_valid(matrix)

    R = valid_matrix[:3, :3]
    t = valid_matrix[:3, 3]

    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t

    return Tinv


def _make_matrix_valid(matrix: NDArrayf) -> NDArrayf:
    """
    Make affine matrix valid given numerical imprecisions.

    :param matrix: Input affine matrix.
    :return: Valid matrix.
    """

    # Copy matrix
    T = np.asarray(matrix).copy()

    # Enforce last row
    T[3, :] = [0, 0, 0, 1]

    # Orthogonalize rotation
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R_ortho = U @ Vt
    # Enforce right-handed system
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    T[:3, :3] = R_ortho

    return T


def _apply_matrix_pts(
    epc: gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    z_name: str = "z",
) -> gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud.

    :param epc: Elevation point cloud.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.

    :return: Transformed elevation point cloud.
    """

    # Apply transformation to X/Y/Z arrays
    tx, ty, tz = _apply_matrix_pts_arr(
        x=epc.geometry.x.values,
        y=epc.geometry.y.values,
        z=epc[z_name].values,
        matrix=matrix,
        centroid=centroid,
        invert=invert,
    )

    # Finally, transform back to a new GeoDataFrame
    transformed_epc = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=tx, y=ty, crs=epc.crs),
        data={z_name: tz},
    )

    return transformed_epc


def translations_rotations_from_matrix(
    matrix: NDArrayf, return_degrees: bool = True
) -> tuple[float, float, float, float, float, float]:
    """
    Extract 3 translations (unit of coordinates) and 3 rotations (degrees or radians) from rigid affine matrix.

    The extracted euler rotations use the extrinsic convention.

    :param matrix: Rigid affine matrix of transformation.
    :param return_degrees: Whether to return rotations in degrees, otherwise radians.

    :return: Translations in the X, Y and Z direction and rotations around the X, Y and Z directions.
    """

    # Extract translations
    t1, t2, t3 = matrix[:3, 3]

    # Get rotations from affine matrix
    rots = _matrix_to_euler(matrix[:3, :3])
    if return_degrees:
        rots = np.rad2deg(np.array(rots))

    # Extract rotations
    alpha1, alpha2, alpha3 = rots

    return t1, t2, t3, alpha1, alpha2, alpha3


def _matrix_to_euler(rotation_matrix: NDArrayf, atol: float = 10e-8) -> tuple[float, float, float]:
    """
    Affine matrix to extrinsic Euler angles.

    :param rotation_matrix: Rotation matrix.

    :return: Euler extrinsic angles in radians (rotations about X, Y and Z).
    """

    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=atol):
        raise ValueError("Matrix is not orthogonal")

    if abs(rotation_matrix[2, 0]) < 1 - atol:
        beta = -np.arcsin(rotation_matrix[2, 0])
        cb = np.cos(beta)

        alpha = np.arctan2(rotation_matrix[2, 1] / cb, rotation_matrix[2, 2] / cb)
        gamma = np.arctan2(rotation_matrix[1, 0] / cb, rotation_matrix[0, 0] / cb)

    # Gimbal lock
    else:
        beta = np.pi / 2 if rotation_matrix[2, 0] <= -1 else -np.pi / 2
        alpha = 0.0
        gamma = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

    return float(alpha), float(beta), float(gamma)


def _iterate_affine_regrid_small_rotations(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    centroid: tuple[float, float, float] | None = None,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
) -> tuple[NDArrayf, rio.transform.Affine]:
    """
    Iterative process to find the best reprojection of affine transformation for small rotations.

    Faster than regridding point cloud by triangulation of points (for instance with scipy.interpolate.griddata).
    """

    # Convert DEM to elevation point cloud, keeping all exact grid coordinates X/Y even for NaNs
    dem_rst = geoutils.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
    epc = dem_rst.to_pointcloud(data_column_name="z", skip_nodata=False).ds

    # Exact affine transform of elevation point cloud (which yields irregular coordinates in 2D)
    tz0 = _apply_matrix_pts_arr(
        x=epc.geometry.x.values, y=epc.geometry.y.values, z=epc.z.values, matrix=matrix, centroid=centroid
    )[2]

    # We need to find the elevation Z of a transformed DEM at the exact grid coordinates X,Y
    # Which means we need to find coordinates X',Y',Z' of the original DEM that, after the exact affine transform,
    # fall exactly on regular X,Y coordinates

    # 1/ The elevation of the original DEM, Z', is simply a 2D interpolator function of X',Y' (bilinear, typically)
    # (We create the interpolator only once here for computational speed, instead of using Raster.interp_points)
    xycoords = dem_rst.coords(grid=False)
    z_interp = scipy.interpolate.RegularGridInterpolator(
        points=(np.flip(xycoords[1], axis=0), xycoords[0]), values=dem, method=resampling, bounds_error=False
    )

    # 2/ As a first guess of a transformed DEM elevation Z near the grid coordinates, we initialize with the elevations
    # of the nearest point from the transformed elevation point cloud

    # OLD METHOD
    # (Longest step computationally)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
    #     nearest = gpd.sjoin_nearest(epc, trans_epc)
    #
    # # In case several points are found at exactly the same distance, take the mean of their elevations
    # new_z = nearest.groupby(by=nearest.index)["z_left"].mean().values

    # NEW METHOD: Use the transformed elevation instead of searching for a nearest neighbour,
    # is close enough for small rotations! (and only creates a couple more iterations instead of a full search)
    new_z = tz0

    # 3/ We then iterate between two steps until convergence:
    # a/ Use the Z guess to derive invert affine transform X',Y' coordinates for the original DEM,
    # b/ Interpolate Z' at new coordinates X',Y' on the original DEM, and apply affine transform to get updated Z guess

    # Start with full array of X/Y regular coordinates (subset during iterations to improve computational efficiency)
    x = epc.geometry.x.values
    y = epc.geometry.y.values

    # Initialize output z array, and array to store points that have converged
    zfinal = np.ones(len(x), dtype=dem.dtype)
    ind_converged = np.zeros(len(x), dtype=bool)

    # For small rotations, and large DEMs (elevation range smaller than the DEM extent), this converges fast
    max_niter = 20  # Maximum iteration number
    niter_check = 5  # Number of iterations between residual checks
    tolerance = 10 ** (-4)  # Tolerance in X/Y relative to resolution of X/Y
    res_x = dem_rst.res[0]  # Resolution in X
    res_y = dem_rst.res[1]  # Resolution in Y
    niter = 1  # Starting iteration

    while niter < max_niter:

        # Invert X,Y (exact grid coordinates) with Z guess to find X',Y' coordinates on original DEM
        tx, ty = _apply_matrix_pts_arr(x=x, y=y, z=new_z, matrix=matrix, invert=True, centroid=centroid)[:2]

        # Interpolate original DEM at X', Y' to get Z', and convert to point cloud
        tz = z_interp((ty, tx))

        # Transform to see if we fall back on our feet (on the regular grid), or if we need to iterate more
        x0, y0, z0 = _apply_matrix_pts_arr(x=tx, y=ty, z=tz, matrix=matrix, centroid=centroid)

        # Only check residuals after first iteration (to remove NaNs) then every 5 iterations to reduce computing time
        if niter == 1 or niter == niter_check:

            # Compute difference between exact grid coordinates and current coordinates, and stop if tolerance reached
            diff_x = x0 - x
            diff_y = y0 - y

            logging.debug(
                "Residual check at iteration number %d:" "\n    Mean diff x: %f" "\n    Mean diff y: %f",
                niter,
                np.nanmean(np.abs(diff_x)),
                np.nanmean(np.abs(diff_y)),
            )

            # Get index of points below tolerance in both X/Y for this subsample (all points before convergence update)
            # Nodata values are considered having converged
            subind_diff_x = np.logical_or(np.abs(diff_x) < (tolerance * res_x), ~np.isfinite(diff_x))
            subind_diff_y = np.logical_or(np.abs(diff_y) < (tolerance * res_y), ~np.isfinite(diff_y))
            subind_converged = np.logical_and(subind_diff_x, subind_diff_y)

            logging.debug(
                "    Points not within tolerance: %d for X; %d for Y",
                np.count_nonzero(~subind_diff_x),
                np.count_nonzero(~subind_diff_y),
            )

            # If all points left are below convergence, update Z one final time and stop here
            if all(subind_converged):
                zfinal[~ind_converged] = z0
                break
            # Otherwise, save Z for new converged points and keep only not converged in next iterations (for speed)
            else:
                zfinal[~ind_converged] = z0
                x = x[~subind_converged]
                y = y[~subind_converged]
                z0 = z0[~subind_converged]

            # Otherwise, for this check, update convergence status for points not having converged yet
            ind_converged[~ind_converged] = subind_converged

        # If another iteration is required, update Z guess and increment
        new_z = z0
        niter += 1

    # 4/ Write the regular-grid point cloud back into a raster
    epc.z = zfinal  # We just replace the Z of the original grid to ensure exact coordinates
    transformed_dem = dem_rst.from_pointcloud_regular(
        epc, transform=transform, shape=dem.shape, data_column_name="z", nodata=-99999
    )

    return transformed_dem.data.filled(np.nan), transform


def invert_matrix(matrix: NDArrayf, atol: float = 10e-8) -> NDArrayf:
    """
    Invert a transformation matrix.

    :param matrix: Affine transformation matrix.

    :return: Inverted transformation matrix.
    """

    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("Not affine")

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError("Not a rigid transform")

    # Make valid before inversion
    valid_matrix = _make_matrix_valid(matrix)

    R = valid_matrix[:3, :3]
    t = valid_matrix[:3, 3]

    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t

    return Tinv


def _make_matrix_valid(matrix: NDArrayf) -> NDArrayf:
    """
    Make affine matrix valid given numerical imprecisions.

    :param matrix: Input affine matrix.
    :return: Valid matrix.
    """

    # Copy matrix
    T = np.asarray(matrix).copy()

    # Enforce last row
    T[3, :] = [0, 0, 0, 1]

    # Orthogonalize rotation
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R_ortho = U @ Vt
    # Enforce right-handed system
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    T[:3, :3] = R_ortho

    return T


def _apply_matrix_pts_arr(
    x: NDArrayf,
    y: NDArrayf,
    z: NDArrayf,
    matrix: NDArrayf,
    centroid: tuple[float, float, float] | None = None,
    invert: bool = False,
) -> tuple[NDArrayf, NDArrayf, NDArrayf]:
    """Apply matrix to points as arrays with array outputs (to improve speed in some functions)."""

    # Invert matrix if required
    if invert:
        matrix = invert_matrix(matrix)

    # First, get 4xN array, adding a column of ones for translations during matrix multiplication
    points = np.vstack([x, y, z, np.ones(len(x))])

    # Temporarily subtract centroid coordinates
    if centroid is not None:
        points[:3, :] -= np.array(centroid)[:, None]

    # Transform using matrix multiplication, and get only the first three columns
    transformed_points = (matrix @ points)[:3, :]

    # Add back centroid coordinates
    if centroid is not None:
        transformed_points += np.array(centroid)[:, None]

    return transformed_points[0, :], transformed_points[1, :], transformed_points[2, :]


def _apply_matrix_pts(
    epc: gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    z_name: str = "z",
) -> gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud.

    :param epc: Elevation point cloud.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.

    :return: Transformed elevation point cloud.
    """

    # Apply transformation to X/Y/Z arrays
    tx, ty, tz = _apply_matrix_pts_arr(
        x=epc.geometry.x.values,
        y=epc.geometry.y.values,
        z=epc[z_name].values,
        matrix=matrix,
        centroid=centroid,
        invert=invert,
    )

    # Finally, transform back to a new GeoDataFrame
    transformed_epc = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=tx, y=ty, crs=epc.crs),
        data={z_name: tz},
    )

    return transformed_epc
