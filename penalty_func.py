import numpy as np


def get_p_width(delta):
    if delta < 0:
        return np.inf
    d = int(delta / 10)
    if d == 0:
        return 0
    elif d <= 3:
        return 1
    elif d <= 6:
        return 2
    elif d <= 9:
        return 3
    elif d <= 12:
        return 5
    elif d <= 15:
        return 10
    elif d <= 18:
        return 20
    elif d <= 21:
        return 30
    elif d <= 24:
        return 50
    elif d <= 27:
        return 70
    elif d <= 30:
        return 90
    elif d <= 33:
        return 120
    elif d <= 36:
        return 150
    elif d <= 54:
        return 200
    elif d <= 90:
        return 500
    elif d <= 150:
        return 1000
    else:
        assert False, f'delta={delta}mm'


h_list = [0, 5, 15, 35, 60, 75]


def get_p_hard(hard):
    h = int(abs(hard))
    return h_list[h]


def get_p_thick(thick):
    """
    former - latter
    :param thick: mm
    :return:
    """
    increase = 0 if thick <= 0 else 1
    t = abs(thick / 10)
    if t < 0.0003:
        p = 0
    elif t <= 0.03:
        p = 3
    elif t <= 0.06:
        p = 7
    elif t <= 0.09:
        p = 12
    elif t <= 0.12:
        p = 18
    elif t <= 0.15:
        p = 25
    elif t <= 0.18:
        p = 33
    elif t <= 0.21:
        p = 42
    elif t <= 0.24:
        p = 52
    elif t <= 0.30:
        p = 66
    elif t <= 0.45:
        p = 99
    elif t <= 3:
        p = 199
    return p*(1+increase)
