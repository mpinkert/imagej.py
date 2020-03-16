import imagej
import pytest

def pytest_addoption(parser):
        parser.addoption(
                "--ij", action="store", default=None, help="directory to IJ"
        )
        parser.addoption(
                "--headless", action="store", default='True', help="Start in headless mode"
        )


@pytest.fixture(scope='session')
def ij_fixture(request):
        ij_dir = request.config.getoption('--ij', default=None)
        headless = bool(request.config.getoption('--headless', default=True))

        ij_wrapper = imagej.init(ij_dir, headless=headless)

        return ij_wrapper
