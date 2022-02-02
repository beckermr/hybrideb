# flake8: noqa
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pizza_cutter_metadetect")
except PackageNotFoundError:
    # package is not installed
    pass

from ._hybrideb import HybridEB, GaussEB, SimpleGaussEB
from ._bineb import BinEB
