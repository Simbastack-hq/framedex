"""framedex — A queryable knowledge base for your video archive."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("framedex")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+unknown"
