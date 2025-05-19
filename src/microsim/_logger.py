import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_log_indent = ContextVar("log_indent", default=0)
logger = logging.getLogger("microsim")


@contextmanager
def logging_indented(delta: int = 1) -> Iterator[None]:
    token = _log_indent.set(_log_indent.get() + delta)
    try:
        yield
    finally:
        _log_indent.reset(token)


class IndentPrefixFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        prefix = "  " * _log_indent.get()
        record.msg = prefix + str(record.msg)
        return True


_filter = IndentPrefixFilter()

try:
    from rich.logging import RichHandler
except ImportError:
    handler = logging.StreamHandler()
    handler.addFilter(_filter)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
else:
    handler = RichHandler()
    handler.addFilter(_filter)

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X:%f]",
    handlers=[handler],
)
