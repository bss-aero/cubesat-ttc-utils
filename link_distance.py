import numpy as np

from constants import *


def angle_from_sides(a, b, c):
    return np.arccos((c ** 2 + b ** 2 - a ** 2) / (2 * b * c))


def get_link_distance(r, elev):
    return -BODY_RADIUS * np.sin(elev) + np.sqrt(r ** 2 - BODY_RADIUS ** 2 * np.cos(elev) ** 2)


def get_link_aperture_body(r, elev):
    link_distance = get_link_distance(r, elev)
    return angle_from_sides(link_distance, BODY_RADIUS, r)


def get_link_aperture_sat(r, elev):
    link_distance = get_link_distance(r, elev)
    return angle_from_sides(BODY_RADIUS, link_distance, r)
