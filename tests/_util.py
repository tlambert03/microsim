import os
import socket
import sys
from collections.abc import Callable

import pytest

# matplotlib's marker/Agg deepcopy recurses infinitely on CPython 3.14 (even a
# bare `ax.plot(...)` raises RecursionError). Skip plotting until matplotlib ships
# a fix. https://github.com/matplotlib/matplotlib
skipif_mpl_py314 = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="matplotlib plotting recurses on CPython 3.14 (upstream bug)",
)

try:
    if os.getenv("MICROSIM_TEST_NO_INTERNET"):
        raise OSError("Skipping internet test due to MICROSIM_TEST_NO_INTERNET")
    socket.create_connection(("8.8.8.8", 53), timeout=1)
    HAVE_INTERNET = True
except OSError:
    HAVE_INTERNET = False


def skipif_no_internet(func: Callable) -> Callable:
    # if there is no internet...

    if not HAVE_INTERNET:
        func = pytest.mark.skip("No internet connection")(func)

    return func
