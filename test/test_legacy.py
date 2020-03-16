import pytest
import numpy

@pytest.fixture(scope="module")
def arr():
    empty_array = numpy.zeros([512, 512])
    return empty_array


class TestIJ1ToIJ2Synchronization(object):
    def testGetImagePlusSynchronizesFromIJ1ToIJ2(self, ij_fixture, arr):
        if not ij_fixture.legacy_enabled:
            pytest.skip("No IJ1.  Skipping test.")

        original = arr[0, 0]
        ds = ij_fixture.py.to_java(arr)
        ij_fixture.ui().show(ds)
        macro = """run("Add...", "value=5");"""
        ij_fixture.py.run_macro(macro)
        imp = ij_fixture.py.get_image_plus()

        assert arr[0, 0] == original + 5

    def testSynchronizeFromIJ1ToNumpy(self, ij_fixture, arr):
        if not ij_fixture.legacy_enabled:
            pytest.skip("No IJ1.  Skipping test.")

        original = arr[0, 0]
        ds = ij_fixture.py.to_dataset(arr)
        ij_fixture.ui().show(ds)
        imp = ij_fixture.py.get_image_plus()
        imp.getProcessor().add(5)
        ij_fixture.py.synchronize_ij2_to_ij1(imp)

        assert arr[0, 0] == original + 5

    def testWindowToNumpyConvertsActiveImageToXarray(self, ij_fixture, arr):
        if not ij_fixture.legacy_enabled:
            pytest.skip("No IJ1.  Skipping test.")

        ds = ij_fixture.py.to_dataset(arr)
        ij_fixture.ui().show(ds)
        new_arr = ij_fixture.py.window_to_xarray()
        assert (arr == new_arr.values).all

    def testFunctionsThrowWarningIfLegacyNotEnabled(self, ij_fixture):
        if ij_fixture.legacy_enabled:
            pytest.skip("IJ1 installed.  Skipping test")

        with pytest.warns(UserWarning):
            ij_fixture.py.synchronize_ij2_to_ij1()
        with pytest.warns(UserWarning):
            ij_fixture.py.get_image_plus()
        with pytest.warns(UserWarning):
            ij_fixture.py.window_to_xarray()
