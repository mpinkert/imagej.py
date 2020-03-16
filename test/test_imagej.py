import numpy as np
import xarray as xr
import pytest


class TestImageJ(object):

    def testFrangi(self, ij_fixture):
        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        result = np.zeros(input_array.shape)
        ij_fixture.op().filter().frangiVesselness(ij_fixture.py.to_java(result), ij_fixture.py.to_java(input_array), [1, 1], 4)
        correct_result = np.array([[0, 0, 0, 0.94282, 0.94283], [0, 0, 0, 0.94283, 0.94283]])
        result = np.ndarray.round(result, decimals=5)
        assert (result == correct_result).all()

    def testGaussian(self, ij_fixture):
        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])

        output_array = ij_fixture.op().filter().gauss(ij_fixture.py.to_java(input_array), 10)
        result = []
        correct_result = [8440, 8440, 8439, 8444]
        ra = output_array.randomAccess()
        for x in [0, 1]:
            for y in [0, 1]:
                ra.setPosition(x, y)
                result.append(ra.get().get())
        assert result==correct_result

    """
    def testTopHat(self, ij_fixture):
        ArrayList = autoclass('java.util.ArrayList')
        HyperSphereShape = autoclass('net.imglib2.algorithm.neighborhood.HyperSphereShape')
        Views = autoclass('net.imglib2.view.Views')

        result = []
        correct_result = [0, 0, 0, 1000, 2000, 4000, 7000, 12000, 20000, 33000]

        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        output_array = np.zeros(input_array.shape)
        java_out = Views.iterable(ij_fixture.py.to_java(output_array))
        java_in = ij_fixture.py.to_java(input_array)
        shapes = ArrayList()
        shapes.add(HyperSphereShape(5))

        ij_fixture.op().morphology().topHat(java_out, java_in, shapes)
        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())

        assert result==correct_result
    """

    def testImageMath(self, ij_fixture):
        from jnius import autoclass
        Views = autoclass('net.imglib2.view.Views')

        input_array = np.array([[1, 1, 2], [3, 5, 8]])
        result = []
        correct_result = [192, 198, 205, 192, 198, 204]
        java_in = Views.iterable(ij_fixture.py.to_java(input_array))
        java_out = ij_fixture.op().image().equation(java_in, "64 * (Math.sin(0.1 * p[0]) + Math.cos(0.1 * p[1])) + 128")

        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())
        assert result==correct_result

    def testPluginsLoadUsingPairwiseStitching(self, ij_fixture):
        if not ij_fixture.legacy_enabled:
            pytest.skip("No IJ1.  Skipping test.")

        macro = """
        newImage("Tile1", "8-bit random", 512, 512, 1);
        newImage("Tile2", "8-bit random", 512, 512, 1);
        """
        plugin = 'Pairwise stitching'
        args = {'first_image': 'Tile1', 'second_image': 'Tile2'}

        ij_fixture.script().run('macro.ij_fixturem', macro, True).get()
        ij_fixture.py.run_plugin(plugin, args)
        from jnius import autoclass
        WindowManager = autoclass('ij_fixture.WindowManager')
        result_name = WindowManager.getCurrentImage().getTitle()

        ij_fixture.script().run('macro.ij_fixturem', 'run("Close All");', True).get()

        assert result_name=='Tile1<->Tile2'


class TestXarrayConversion(object):
    def testCstyleArrayWithLabeledDimsConverts(self, ij_fixture):
        xarr = xr.DataArray(np.random.rand(5, 4, 3, 6, 12), dims=['T', 'Z', 'C', 'Y', 'X'],
                             coords={'X': range(0, 12), 'Y': np.arange(0, 12, 2), 'C': ['R', 'G', 'B'],
                                     'Z': np.arange(10, 50, 10), 'T': np.arange(0, 0.05, 0.01)},
                             attrs={'Hello': 'Wrld'})

        dataset = ij_fixture.py.to_java(xarr)
        from jnius import cast
        axes = [cast('net.imagej.axis.DefaultLinearAxis', dataset.axis(axnum)) for axnum in range(5)]
        labels = [axis.type().getLabel() for axis in axes]
        origins = [axis.origin() for axis in axes]
        scales = [axis.scale() for axis in axes]

        assert origins==[0, 0, 0, 10, 0]
        assert scales==[1, 2, 1, 10, 0.01]

        assert list(reversed(xarr.dims))==labels

        assert xarr.attrs==ij_fixture.py.from_java(dataset.getProperties())

    def testFstyleArrayWiathLabeledDimsConverts(self, ij_fixture):
        xarr = xr.DataArray(np.ndarray([5, 4, 3, 6, 12], order='F'), dims=['t', 'z', 'c', 'y', 'x'],
                            coords={'x': range(0, 12), 'y': np.arange(0, 12, 2),
                                    'z': np.arange(10, 50, 10), 't': np.arange(0, 0.05, 0.01)},
                            attrs={'Hello': 'Wrld'})

        dataset = ij_fixture.py.to_java(xarr)
        from jnius import cast
        axes = [cast('net.imagej.axis.DefaultLinearAxis', dataset.axis(axnum)) for axnum in range(5)]
        labels = [axis.type().getLabel() for axis in axes]
        origins = [axis.origin() for axis in axes]
        scales = [axis.scale() for axis in axes]

        assert origins==[0, 10, 0, 0, 0]
        assert scales==[0.01, 10, 1, 2, 1]

        assert [dim.upper() for dim in xarr.dims]==labels
        assert xarr.attrs==ij_fixture.py.from_java(dataset.getProperties())

    def testDatasetConvertsToXarray(self, ij_fixture):
        xarr = xr.DataArray(np.random.rand(5, 4, 3, 6, 12), dims=['t', 'z', 'c', 'y', 'x'],
                             coords={'x': list(range(0, 12)), 'y': list(np.arange(0, 12, 2)), 'c': [0, 1, 2],
                                     'z': list(np.arange(10, 50, 10)), 't': list(np.arange(0, 0.05, 0.01))},
                             attrs={'Hello': 'Wrld'})

        dataset = ij_fixture.py.to_java(xarr)

        invert_xarr = ij_fixture.py.from_java(dataset)
        assert (xarr.values == invert_xarr.values).all()

        assert list(xarr.dims)==list(invert_xarr.dims)
        for key in xarr.coords:
            assert (xarr.coords[key] == invert_xarr.coords[key]).all()
        assert xarr.attrs==invert_xarr.attrs

