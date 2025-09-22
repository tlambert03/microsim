import socket
from collections.abc import Callable

import pytest

try:
    socket.create_connection(("8.8.8.8", 53), timeout=1)
    HAVE_INTERNET = True
except OSError:
    HAVE_INTERNET = False


def skipif_no_internet(func: Callable) -> Callable:
    # if there is no internet...

    if not HAVE_INTERNET:
        func = pytest.mark.skip("No internet connection")(func)

    return func
