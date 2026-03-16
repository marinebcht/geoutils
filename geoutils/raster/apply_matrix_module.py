from __future__ import annotations

import copy
import inspect
import logging
import warnings
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    TypedDict,
    TypeVar,
    overload,
)

import affine
import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import rasterio as rio
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.interface.interpolation import _interp_points_base
from geoutils.raster.referencing import _cast_pixel_interpretation, _coords
from geoutils.raster.transformation import _translate, _build_geotiling_and_meta_apply_matrix
from geoutils.multiproc.mparray import MultiprocConfig, _write_multiproc_result

from geoutils.raster.apply_matrix_function import *

from geoutils.multiproc.chunked import (
    ChunkedGeoGrid,
    GeoGrid,
    _chunks2d_from_chunksizes_shape,
)

@overload
def _reproject_horizontal_shift_samecrs(
    raster_arr: NDArrayf,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine = None,
    *,
    return_interpolator: Literal[False] = False,
    resampling: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> NDArrayf: ...


@overload
def _reproject_horizontal_shift_samecrs(
    raster_arr: NDArrayf,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine = None,
    *,
    return_interpolator: Literal[True],
    resampling: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf]: ...


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
    if not return_interpolator:
        coords_dst = _coords(transform=dst_transform, area_or_point="Area", shape=raster_arr.shape)
        # Flatten the arrays (only 1D supported in rowcol/xy after Rasterio 1.4)
        coords_dst = (coords_dst[0].ravel(), coords_dst[1].ravel())
    # If we just want the interpolator, we don't need to coordinates of destination points
    else:
        coords_dst = None

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
    transform: rio.transform.Affine = None,
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
    # dask_backend = da is not None and elev._chunks is not None
    dask_backend = False

    # Apply matrix to elevation point cloud
    if isinstance(elev, gpd.GeoDataFrame):
        return _apply_matrix_pts(epc=elev, matrix=matrix, invert=invert, centroid=centroid, z_name=z_name)
    # Or apply matrix to raster (often requires re-gridding)
    else:



        # If using Multiprocessing backend, process and return None (files written on disk)
        if mp_backend:
            # Get depth of overlap
            depth = 10  # ath.ceil(depth)
            _multiproc_apply_matrix(elev, multiproc_config, transform, matrix, invert, centroid, resampling)
        else :
            # First, we apply the affine matrix for the array/transform
            if isinstance(elev, gu.Raster):
                transform = elev.transform
                dem = elev.data.filled(np.nan)
            else:
                dem = elev

            applied_dem, out_transform = _apply_matrix_rst(
                dem=dem,
                transform=transform,
                matrix=matrix,
                invert=invert,
                centroid=centroid,
                resampling=resampling,
                **kwargs,
            )

            # Then, if resample is True, we reproject the DEM from its out_transform onto the transform
            if resample:
                applied_dem = _reproject_horizontal_shift_samecrs(
                    applied_dem, src_transform=out_transform, dst_transform=transform, resampling=resampling
                )
                out_transform = transform

            # We return a raster if input was a raster
            if isinstance(elev, gu.Raster):
                applied_dem = gu.Raster.from_array(applied_dem, out_transform, elev.crs, elev.nodata)
                return applied_dem
            return applied_dem, out_transform


def _multiproc_apply_matrix(
    rst: Raster,
    mp_config: MultiprocConfig,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    force_regrid_method: Literal["iterative", "griddata"] | None = None,
) -> tuple[NDArrayf, rio.transform.Affine]:
    print ("_multiproc_apply_matrix")

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_chunks = _chunks2d_from_chunksizes_shape(
        chunksizes=(mp_config.chunk_size, mp_config.chunk_size), shape=rst.shape
    )
    print (src_chunks)
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta_apply_matrix(
            src_count=rst.count,
            src_shape=rst.shape,
            src_transform=rst.transform,
            src_crs=rst.crs,
            dst_shape=rst.shape,
            dst_transform=rst.transform,
            dst_crs=rst.crs,
            src_chunks=src_chunks,
            dst_chunksizes=(mp_config.chunk_size, mp_config.chunk_size),
        )
    )


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
    dem_rst = gu.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
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

    # 1/ Check if the matrix only contains a Z correction, in that case only shift the DEM values by the vertical shift
    if np.array_equal(shift_z_only_matrix, matrix) and force_regrid_method is None:
        return dem + matrix[2, 3], transform

    # 2/ Check if the matrix contains only translations, in that case only shift the DEM only by translation
    if np.array_equal(shift_only_matrix, matrix) and force_regrid_method is None:
        new_transform = _translate(transform, xoff=matrix[0, 3], yoff=matrix[1, 3])
        return dem + matrix[2, 3], new_transform

    # 3/ If matrix contains only small rotations (less than 20 degrees), use the fast iterative reprojection
    rotations = translations_rotations_from_matrix(matrix)[3:]
    if all(np.abs(rot) < 20 for rot in rotations) and force_regrid_method is None or force_regrid_method == "iterative":
        new_dem, transform = _iterate_affine_regrid_small_rotations(
            dem=dem, transform=transform, matrix=matrix, centroid=centroid, resampling=resampling
        )
        return new_dem, transform

    # 4/ Otherwise, use a delauney triangulation interpolation of the transformed point cloud
    # Convert DEM to elevation point cloud, keeping all exact grid coordinates X/Y even for NaNs
    dem_rst = gu.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
    epc = dem_rst.to_pointcloud(data_column_name="z").ds
    trans_epc = _apply_matrix_pts(epc, matrix=matrix, centroid=centroid)

    new_dem = _grid_pointcloud(
        trans_epc, grid_coords=dem_rst.coords(grid=False), data_column_name="z", resampling=resampling
    )[0]

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
