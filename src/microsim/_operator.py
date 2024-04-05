from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from microsim.models._renderable import Renderable


def convolve(a, b) -> "Renderable": ...
