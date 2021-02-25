from typing import Tuple


def hex2float(v: str) -> Tuple[float, float, float]:
    if v.startswith("#"):
        v = v[1:]

    r, g, b = v[0:2], v[2:4], v[4:6]
    r, g, b = int(r, 16), int(g, 16), int(b, 16)
    r, g, b = r / 255, g / 255, b / 255
    return r, g, b
