import geoutils as gu
import numpy as np
import pytest

from geoutils.multiproc.mparray import MultiprocConfig, _write_multiproc_result
import rasterio as rio
from geoutils import open_raster

from pathlib import Path


def _euler_to_matrix(alpha1: float, alpha2: float, alpha3: float):
    """
    Extrinsic Euler angles to affine matrix.

    :param alpha1: Angle around X in radians.
    :param alpha2: Angle around Y in radians.
    :param alpha3: Angle around Z in radians.

    :return: Rotation matrix.
    """

    def Rx(a: float):
        ca, sa = np.cos(a), np.sin(a)
        return np.array(
            [
                [1, 0, 0],
                [0, ca, -sa],
                [0, sa, ca],
            ]
        )

    def Ry(a: float):
        ca, sa = np.cos(a), np.sin(a)
        return np.array(
            [
                [ca, 0, sa],
                [0, 1, 0],
                [-sa, 0, ca],
            ]
        )

    def Rz(a: float):
        ca, sa = np.cos(a), np.sin(a)
        return np.array(
            [
                [ca, -sa, 0],
                [sa, ca, 0],
                [0, 0, 1],
            ]
        )

    return Rz(alpha3) @ Ry(alpha2) @ Rx(alpha1)


def matrix_from_translations_rotations(
    t1: float = 0.0,
    t2: float = 0.0,
    t3: float = 0.0,
    alpha1: float = 0.0,
    alpha2: float = 0.0,
    alpha3: float = 0.0,
    use_degrees: bool = True,
):
    """
    Build rigid affine matrix based on 3 translations (unit of coordinates) and 3 rotations (degrees or radians).

    The euler rotations use the extrinsic convention.

    :param t1: Translation in the X (west-east) direction (unit of coordinates).
    :param t2: Translation in the Y (south-north) direction (unit of coordinates).
    :param t3: Translation in the Z (vertical) direction (unit of DEM).
    :param alpha1: Rotation around the X (west-east) direction.
    :param alpha2: Rotation around the Y (south-north) direction.
    :param alpha3: Rotation around the Z (vertical) direction.
    :param use_degrees: Whether to use degrees for input rotations, otherwise radians.

    :raises ValueError: If the given translation or rotations contained invalid values.

    :return: Rigid affine matrix of transformation.
    """

    # Initialize diagonal matrix
    matrix = np.eye(4)
    # Convert euler angles to rotation matrix
    e = np.array([alpha1, alpha2, alpha3])
    # If angles were given in degrees
    if use_degrees:
        e = np.deg2rad(e)
    rot_matrix = _euler_to_matrix(alpha1=e[0], alpha2=e[1], alpha3=e[2])

    # Add rotation matrix, and translations
    matrix[0:3, 0:3] = rot_matrix
    matrix[:3, 3] = [t1, t2, t3]

    return matrix


class TestApplyMatrixManipulation:
    pytest.importorskip("dask")

    # Identity transformation
    matrix_identity = np.diag(np.ones(4, float))

    # Vertical shift
    matrix_vertical = matrix_identity.copy()
    matrix_vertical[2, 3] = 1

    # Vertical and horizontal shifts
    matrix_translations = matrix_identity.copy()
    matrix_translations[:3, 3] = [0.5, 1, 1.5]

    # Single rotation
    rotation = np.deg2rad(5)
    matrix_rotations = matrix_identity.copy()
    matrix_rotations[1, 1] = np.cos(rotation)
    matrix_rotations[2, 2] = np.cos(rotation)
    matrix_rotations[2, 1] = -np.sin(rotation)
    matrix_rotations[1, 2] = np.sin(rotation)

    # Mix of translations and rotations in all axes (X, Y, Z) simultaneously
    rotation_x = 5
    rotation_y = 10
    rotation_z = 3
    e = np.deg2rad(np.array([rotation_x, rotation_y, rotation_z]))
    trans_x = 0.5
    trans_y = 1
    trans_z = 1.5
    # This is a 3x3 rotation matrix
    matrix_all = matrix_from_translations_rotations(
        trans_x, trans_y, trans_z, rotation_x, rotation_y, rotation_z
    )
    list_matrices = [(0, matrix_identity), (1, matrix_vertical), (2, matrix_translations),
                     (3, matrix_rotations), (4, matrix_all)]

    @pytest.mark.parametrize("path_index", [0])  # todo ?
    @pytest.mark.parametrize("matrix", list_matrices)
    @pytest.mark.parametrize("chunk_size", [5, 8, 12])
    @pytest.mark.parametrize("invert", [False, True])
    @pytest.mark.parametrize("resampling", [None, "nearest", "linear", "cubic", "quintic"])
    def test_apply_matrix_dask_multi(
        self,
        matrix,
        path_index,
        chunk_size: int,
        invert: bool,
        resampling: str,
        tmp_path: Path,
        lazy_test_files_tiny: list[str],
    ) -> None:
        import dask.array as da

        # Base raster input (in-memory)
        """dem_arr = np.linspace(0, 99, 100).reshape(10, 10)
        transform = rio.transform.from_origin(0, 5, 1, 1)
        raster_base = gu.Raster.from_array(dem_arr, transform=transform, crs=4326, nodata=200)
        assert raster_base.is_loaded
        raster_base.to_file(tmp_path / "raster_base.tif")"""

        # 1/ Prepare backend inputs
        # Get filepath of on-disk (for laziness) test file
        path_raster = lazy_test_files_tiny[path_index]

        # Base raster input (in-memory)
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        assert raster_base.is_loaded

        # Base data array input (in-memory)
        ds_base = open_raster(path_raster)
        ds_base.load()
        assert ds_base._in_memory

        # Multiprocessing input (lazy)
        raster_mp = gu.Raster(path_raster)
        assert not raster_mp.is_loaded

        # Dask input (lazy)
        ds_dask = open_raster(path_raster, chunks={"x": chunk_size, "y": chunk_size})
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert ds_dask.data.chunks is not None

        # Get centroid and resample info
        epc = raster_base.to_pointcloud(data_column_name="z").ds
        centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)
        resample = resampling is not None
        if resample is False:
            resampling = "nearest"

        # Run apply_matrix for each backend
        print ("# run base")
        base_am = gu.raster.apply_matrix_module.apply_matrix(
            raster_base, matrix[1], invert=invert, centroid=centroid, resample=resample, resampling=resampling
        )
        # Valid classique apply_matrix
        if resample is False:
            path = str(matrix[0]) + "_" + str(invert) + "_" + str(resample) + "_None.tif"
        else:
            path = str(matrix[0]) + "_" + str(invert) + "_" + resampling + ".tif"

        path = "/home/mbouchet/Documents/xDem_project/new_xdem/xdem/tmp/" + path
        dem_ref_xdem = gu.Raster(path)
        dem_ref_xdem.load()
        assert isinstance(base_am, gu.Raster)
        assert base_am.nodata == dem_ref_xdem.nodata
        assert base_am.crs == dem_ref_xdem.crs
        assert base_am.transform == dem_ref_xdem.transform
        assert np.all(base_am.get_mask() == dem_ref_xdem.get_mask())
        assert np.all(np.array(base_am.data - dem_ref_xdem.data)[base_am.get_mask() == False] < 10e-2)

        diff = 10e-5

        print("# run multi")

        # Multiprocessing config
        multiproc_config = MultiprocConfig(chunk_size=chunk_size, outfile=tmp_path / "multi.tif")

        mp_am = gu.raster.apply_matrix_module.apply_matrix(
            raster_base,
            matrix[1],
            invert=invert,
            centroid=centroid,
            resample=resample,
            resampling=resampling,
            multiproc_config=multiproc_config,
        )

        # 4/ Laziness checks
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert not raster_mp.is_loaded

        # 5/ Output checks: all backends must match base

        # Multi
        assert isinstance(mp_am, gu.Raster)
        assert mp_am.nodata == base_am.nodata
        # assert mp_am.dtype == type(matrix[0,0])
        assert mp_am.crs == base_am.crs
        assert mp_am.transform == base_am.transform
        assert np.all(mp_am.get_mask() == base_am.get_mask())
        assert np.all(mp_am.get_mask() == base_am.get_mask())



        assert np.all(np.array(base_am.data - mp_am.data)[base_am.get_mask() == False] < diff)

        # Dask
        print("# run dask")

        dask_am = gu.raster.apply_matrix_module.apply_matrix(
            ds_dask.rst, matrix[1], invert=invert, centroid=centroid, resample=resample, resampling=resampling
        )

        assert not dask_am._in_memory
        dask_am = dask_am.compute()
        assert dask_am._in_memory
        assert dask_am.rst.nodata == base_am.nodata
        assert dask_am.rst.dtype == base_am.dtype
        assert dask_am.rst.crs == base_am.crs
        assert dask_am.rst.transform == base_am.transform

        assert np.all(np.isnan(dask_am.rst.data[base_am.get_mask()]))
        assert np.all(np.array(base_am.data - dask_am.rst.data)[base_am.get_mask() == False] < diff)



def test__rio_reproject():
    dst_arr_after_apply_matrix = np.linspace(0, 99, 100).reshape(10, 10)
    print(dst_arr_after_apply_matrix)

    from pyproj import CRS
    from affine import Affine
    from geoutils.raster.transformation import _rio_reproject

    def run(array, src_transform, dst_transform):
        kwargs = {
            "dst_shape": (5, 5),
            "src_transform": src_transform,
            "dst_transform": dst_transform,
            "dtype": np.float64,
            "num_threads": 1,
            "src_nodata": 200,
            "dst_nodata": 200,
            "src_crs": CRS.from_epsg(4326),
            "dst_crs": CRS.from_epsg(4326),
        }

        dst_arr_res = _rio_reproject(src_arr=array, reproj_kwargs=kwargs)  # type: ignore
        return dst_arr_res

    # upper left
    src_transform = Affine(1.0, 0.0, np.float64(0.0), 0.0, -1.0, np.float64(5.0))
    dst_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
    print(run(dst_arr_after_apply_matrix, src_transform, dst_transform))

    # upper right
    src_transform = Affine(1.0, 0.0, np.float64(0.0), 0.0, -1.0, np.float64(5.0))
    dst_transform = Affine(1.0, 0.0, 5.0, 0.0, -1.0, 5.0)
    print(run(dst_arr_after_apply_matrix, src_transform, dst_transform))

    # bottom left
    src_transform = Affine(1.0, 0.0, np.float64(0.0), 0.0, -1.0, np.float64(5.0))
    dst_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    print(run(dst_arr_after_apply_matrix, src_transform, dst_transform))

    # bottom right
    src_transform = Affine(1.0, 0.0, np.float64(0.0), 0.0, -1.0, np.float64(5.0))
    dst_transform = Affine(1.0, 0.0, 5.0, 0.0, -1.0, 0.0)
    print(run(dst_arr_after_apply_matrix, src_transform, dst_transform))

    print(l)


def test_repro():
    src_arrs = np.linspace(0, 99, 100).reshape(10, 10)
    transform = rio.transform.from_origin(0, 5, 1, 1)
    chunk_size = 4
    dem = gu.Raster.from_array(src_arrs, transform=transform, crs=4326, nodata=200)

    # Test Normal
    dem_reproj = dem.reproject(res=(1, 1))  # , mp_config=multiproc_config)
    print("Res normal: \n", dem_reproj.data)

    # Test Multi
    multiproc_config = MultiprocConfig(chunk_size=chunk_size, outfile="tmp/test.tif")
    dem_reproj = dem.reproject(res=(1, 1), mp_config=multiproc_config)
    print("Res Multiprocess: \n", dem_reproj.data)

    # Test Dask
    ds = open_raster("input.tif", chunks={"band": 1, "x": chunk_size, "y": chunk_size})
    print(ds.rst.get_stats())
    ds.rst.data.set_mask(src_arrs > 0)

    dem_reproj = ds.rst.reproject(res=(1, 1))
    dem_reproj = dem_reproj.compute()
    print("Res Dask: \n", dem_reproj.rst.data)

    print(l)


def test__regulargrid():

    import scipy

    points = (([3098570.0, 3098540.0, 3098510.0, 3098480.0]), ([489340.0, 489370.0, 489400.0, 489430.0]))
    dem = [
        [251.0, 255.0, 251.0, 255.0],
        [249.0, 253.0, 251.0, 255.0],
        [244.0, 234.0, 251.0, 255.0],
        [235.0, 227.0, 283.0, np.nan],
    ]
    resampling = "cubic"

    z_interp = scipy.interpolate.RegularGridInterpolator(
        points=points,
        values=dem,
        method=resampling,
        bounds_error=False,
        fill_value=None,
    )

"""def test_regulargrid():
    from shapely import Polygon
    coords = ((489700, 3098270), (489700, 3098570), (489340, 3098570), (489340, 3098270), (489700, 3098270))
    polygon = Polygon(coords)
    print (polygon)
    print (l)"""