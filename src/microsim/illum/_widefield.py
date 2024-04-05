from microsim.models import Illumination


class Widefield(Illumination):
    x_offset: float = 0
    y_offset: float = 0
    sigma: float = 1000
