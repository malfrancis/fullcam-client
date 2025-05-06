"""
FullCAM API Client - Python library for interacting with the FullCAM APIs

This library handles:
1. Loading XML configurations from files
2. Sending XML to the FullCAM Plot API
3. Processing CSV responses
4. Converting results to Arrow format for compatibility with various dataframe libraries
5. Managing multiple simulations for comparison and visualization
"""

from importlib.metadata import PackageNotFoundError, version

from fullcam_client.client import FullCAMClient
from fullcam_client.exceptions import FullCAMAPIError, FullCAMClientError
from fullcam_client.simulation import Simulation

try:
    __version__ = version("fullcam-client")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["FullCAMClient", "FullCAMClientError", "FullCAMAPIError", "Simulation"]
