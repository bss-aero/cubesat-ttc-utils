import numpy as np
from scipy.optimize import fsolve

from constants import *
from contact import get_angle
from link_distance import get_link_aperture_body


def get_orbit_normal_vec(lat, inc):
    if (lat < inc) if (inc <= np.pi / 2) else (lat < np.pi - inc):
        return np.array([
            -np.cos(inc) * np.tan(lat),
            np.sqrt(1 - np.cos(inc) ** 2 * (1 + np.tan(lat) ** 2)),
            np.cos(inc),
        ])
    else:
        return np.array([
            -np.sin(inc),
            0,
            np.cos(inc),
        ])


def align(gs, sc):
    orbit_normal = get_orbit_normal_vec(gs.lat, sc.inc)
    new_gs = gs.as_copy(lon=0)

    raan = np.cross(np.array([0, 0, 1]), orbit_normal)
    new_sc1 = sc.as_copy(raan=np.degrees(np.arctan2(raan[1], raan[0])))
    r = np.linalg.inv(new_sc1.rot).dot(np.array([np.cos(gs.lat), 0, np.sin(gs.lat)]))
    f0 = np.arctan2(r[1], r[0])
    new_sc1.f0 = f0
    new_sc1.update()

    raan = np.cross(np.array([0, 0, 1]), orbit_normal * np.array([1, -1, 1]))
    new_sc2 = sc.as_copy(raan=np.degrees(np.arctan2(raan[1], raan[0])))
    r = np.linalg.inv(new_sc2.rot).dot(np.array([np.cos(gs.lat), 0, np.sin(gs.lat)]))
    f0 = np.arctan2(r[1], r[0])
    new_sc2.f0 = f0
    new_sc2.update()

    best_sc = max([new_sc1, new_sc2], key=lambda sc_i: np.linalg.norm(sc_i.get_local_position(0)))

    return new_gs, best_sc


def rotate(r, axis, theta):
    return r * np.cos(theta) + \
           np.cross(axis, r, axis=0) * np.sin(theta) + \
           axis * np.sum(axis * r, axis=0) * (1 - np.cos(theta))


def get_aperture(t, axis_e, axis_s, r, w_e, w_s, target):
    return get_angle(rotate(r, axis_e, t * w_e), rotate(r, axis_s, t * w_s)) - target


def get_max_equatorial_comm_window(sma, min_elev):
    return 2 * get_link_aperture_body(sma, min_elev) / np.abs(np.sqrt(BODY_MU / sma ** 3) - 2 * np.pi / BODY_PERIOD)


def get_max_comm_window(sma, lat, inc, min_elev):
    assert (lat <= inc) if (inc <= np.pi / 2) else (lat <= np.pi - inc)
    axis_e = np.array([[0, 0, 1]]).T
    axis_s = get_orbit_normal_vec(lat, inc).reshape((3, 1))
    r = np.array([[np.cos(lat), 0, np.sin(lat)]]).T
    w_e = 2 * np.pi / BODY_PERIOD
    w_s = np.sqrt(BODY_MU / sma ** 3)
    target = get_link_aperture_body(sma, min_elev)
    return 2 * fsolve(
        get_aperture,
        get_max_equatorial_comm_window(sma, min_elev) / 2,
        (axis_e, axis_s, r, w_e, w_s, target)
    )[0]


def get_aperture_numeric(u, gs, sc):
    gs_th, sc_th, has_sight = gs.has_sight(sc, u)
    target = get_link_aperture_body(np.linalg.norm(sc_th.position, axis=0), gs.min_elev)
    return get_angle(gs_th.position, sc_th.position) - target


def get_max_comm_window_numeric(gs, sc):
    gs_mod, sc_mod = align(gs, sc)
    start = gs_mod.has_sight(sc_mod, 0)
    if (np.linalg.norm(start[1].position) <= BODY_RADIUS) or (not np.all(start[2])):
        return 0

    t_guess = get_max_equatorial_comm_window(sc_mod.sma, gs_mod.min_elev) / 2
    u_guess = fsolve(
        lambda u, target: sc_mod.get_time(u) - target,
        np.array(2 * np.pi * t_guess / sc_mod.period),
        t_guess
    )[0]

    u1 = fsolve(get_aperture_numeric, np.array(u_guess), (gs_mod, sc_mod))[0]
    u0 = fsolve(get_aperture_numeric, np.array(-u_guess), (gs_mod, sc_mod))[0]

    return abs(sc_mod.get_time(u1) - sc_mod.get_time(u0))
