import numpy as np

from constants import *
from link_distance import angle_from_sides

INTEGRATION_RESOLUTION = 1024


def to_db(x):
    return 10 * np.log10(x)


def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def get_angle(v1, v2):
    return np.arccos(np.sum((v1 / np.linalg.norm(v1, axis=0)) * (v2 / np.linalg.norm(v2, axis=0)), axis=0))


class Spacecraft(object):
    def __init__(self, sma, ecc, inc, raan, aop, f0):
        self.sma = sma
        self.ecc = ecc
        self.inc = np.radians(inc)
        self.raan = np.radians(raan)
        self.aop = np.radians(aop)
        self.f0 = np.radians(f0)

        self.n = None
        self.u0 = None
        self.t0 = 0
        self.rot = None
        self.period = None
        self.antenna = None
        self.update()

    def update(self):
        self.n = np.sqrt(BODY_MU / self.sma ** 3)
        self.t0 = 0
        self.u0 = np.arctan2(np.sqrt(1 - self.ecc ** 2) * np.sin(self.f0), self.ecc + np.cos(self.f0))
        self.t0 = self.get_time(0)
        self.rot = rot_z(-self.raan).dot(rot_x(-self.inc)).dot(rot_z(-self.aop))
        self.period = self.get_time(2 * np.pi)

    def set_antenna(self, antenna):
        self.antenna = antenna

    def get_local_position(self, u):
        if not isinstance(u, np.ndarray):
            u = np.array(u)

        return np.array([
            self.sma * (np.cos(u + self.u0) - self.ecc),
            self.sma * np.sin(u + self.u0) * (1 - self.ecc ** 2) ** .5,
            np.zeros(u.shape)
        ])

    def get_global_position(self, u):
        return self.rot.dot(self.get_local_position(u))

    def get_time(self, u):
        if not isinstance(u, np.ndarray):
            u = np.array(u)

        return (u + self.u0 - self.ecc * np.sin(u + self.u0)) / self.n - self.t0

    def as_copy(self, **kwargs):
        args = {
            'sma': self.sma,
            'ecc': self.ecc,
            'inc': np.degrees(self.inc),
            'raan': np.degrees(self.raan),
            'aop': np.degrees(self.aop),
            'f0': np.degrees(self.f0)
        }
        args.update(kwargs)
        result = Spacecraft(**args)
        result.set_antenna(self.antenna)
        return result


class GroundStation(object):
    def __init__(self, lat, lon, min_elev):
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)
        self.min_elev = np.radians(min_elev)
        self.antenna = None

    def set_antenna(self, antenna):
        self.antenna = antenna

    def get_effective_longitude(self, time):
        return 2 * np.pi * (time / BODY_PERIOD % 1) + self.lon

    def get_global_position(self, time):
        if not isinstance(time, np.ndarray):
            time = np.array(time)

        lon = self.get_effective_longitude(time)
        phi = np.ones(lon.shape) * (np.pi / 2 - self.lat)
        return BODY_RADIUS * np.array([
            np.cos(lon) * np.sin(phi),
            np.sin(lon) * np.sin(phi),
            np.cos(phi)
        ])

    def has_sight(self, sc, u):
        time = sc.get_time(u)
        gs_pos = self.get_global_position(time)
        sc_pos = sc.get_global_position(u)
        angles = get_angle(gs_pos, sc_pos - gs_pos)

        return (
            TimeHistory(self, time, u, gs_pos),
            TimeHistory(sc, time, u, sc_pos),
            angles <= np.pi / 2 - self.min_elev
        )

    def as_copy(self, **kwargs):
        args = {
            'lat': np.degrees(self.lat),
            'lon': np.degrees(self.lon),
            'min_elev': np.degrees(self.min_elev)
        }
        args.update(kwargs)
        result = GroundStation(**args)
        result.set_antenna(self.antenna)
        return result


class TimeHistory(object):
    def __init__(self, parent, time, u, position):
        self.parent = parent
        self.time = time
        self.u = u
        self.position = position

    def __len__(self):
        return len(self.time)

    def __getitem__(self, item):
        return TimeHistory(self.parent, self.time[item], self.u[item], self.position[:, item])


class Contact(object):
    def __init__(self, gs, sc, partial):
        self.gs = gs
        self.sc = sc
        self.partial = partial

        self.start = sc.time[0]
        self.end = sc.time[-1]
        self.duration = self.end - self.start

        self._elevation = None
        self._azimuth = None
        self._tracking = None
        self._power = None

    def update(self):
        sc_pos_local = self.sc.position - self.gs.position
        gs_pos_unit = self.gs.position / np.linalg.norm(self.gs.position, axis=0)

        self._elevation = get_angle(self.gs.position, sc_pos_local)
        sc_pos_plane = sc_pos_local - np.linalg.norm(sc_pos_local, axis=0) * np.cos(self._elevation) * gs_pos_unit

        east_theta = np.pi / 2 + self.gs.parent.get_effective_longitude(self.gs.time)
        east = np.array([
            np.cos(east_theta),
            np.sin(east_theta),
            np.zeros(east_theta.shape)
        ])
        north = np.cross(gs_pos_unit, east, axis=0)

        azimuth_north = get_angle(north, sc_pos_plane)
        azimuth_east = get_angle(east, sc_pos_plane)

        self._azimuth = np.where(azimuth_east <= np.pi / 2, azimuth_north, -azimuth_north)

        if (self.gs.parent.antenna is not None) and (self.sc.parent.antenna is not None):
            gs_dist = np.linalg.norm(self.gs.position, axis=0)
            sc_dist = np.linalg.norm(self.sc.position, axis=0)
            sc_dist_local = np.linalg.norm(sc_pos_local, axis=0)

            self._power = self.sc.parent.antenna.transmit(
                self.gs.parent.antenna,
                sc_dist_local,
                angle_from_sides(gs_dist, sc_dist, sc_dist_local),
                0
            )

            deltas = 2 * np.ones(self._azimuth.shape)
            deltas[0] = 1
            deltas[-1] = 1

            diff_az = deltas * np.gradient(self._azimuth)
            diff_az_compl = diff_az - np.sign(diff_az) * 2 * np.pi
            gradient_az = np.where(abs(diff_az) > np.pi, diff_az_compl, diff_az) / (deltas * np.gradient(self.sc.time))
            gradient_elev = np.gradient(self._elevation, self.sc.time)

            self._tracking = np.stack((gradient_az, gradient_elev))

    @property
    def elevation(self):
        if self._elevation is None:
            self.update()
        return np.pi / 2 - self._elevation

    @property
    def azimuth(self):
        if self._azimuth is None:
            self.update()
        return self._azimuth

    @property
    def power(self):
        if self._power is None:
            self.update()
        return self._power

    @property
    def tracking(self):
        if self._tracking is None:
            self.update()
        return self._tracking

    @staticmethod
    def get_contacts(u, gs, sc, partials=False):
        th_gs, th_sc, has_sight = gs.has_sight(sc, u)
        edges = list(np.where(has_sight[:-1] != has_sight[1:])[0])

        start_edge = bool(has_sight[0])
        end_edge = bool(has_sight[-1])

        if start_edge:
            if partials:
                edges.insert(0, 0)
            else:
                edges.pop(0)

        if partials and end_edge:
            edges.append(None)

        contacts = list()

        for limits in zip(edges[::2], edges[1::2]):
            sliced = slice(*limits)
            partial = (limits[0] == 0) or (limits[1] is None)
            contacts.append(
                Contact(th_gs[sliced], th_sc[sliced], partial)
            )

        return contacts


class Antenna(object):
    def __init__(self, freq, vswr=1, radiation_efficiency=1, polarization_efficiency=1, radiation_pattern=None,
                 transmit_pwr=None, sensibility=None):
        assert radiation_efficiency <= 1
        self.freq = freq
        self.vswr = vswr
        self.radiation_efficiency = radiation_efficiency
        self.reflection_efficiency = (1 - ((vswr - 1) / (vswr + 1)) ** 2)
        self.polarization_efficiency = polarization_efficiency
        self.losses = to_db(self.radiation_efficiency * self.reflection_efficiency * self.polarization_efficiency)
        self.transmit_pwr = transmit_pwr
        self.sensibility = sensibility

        self._radiation_pattern = radiation_pattern
        self._directivity = None
        self._gain = None
        self.hpbw = None
        self.p_rad = None
        self.g0 = 0

        if radiation_pattern is not None:
            self.update()

    def update(self, res=INTEGRATION_RESOLUTION):
        elev = np.linspace(0, np.pi, res)
        self.p_rad = 2 * np.pi * np.trapz(self._radiation_pattern(elev) * np.sin(elev), elev)

        self.g0 = self.gain(0)
        target = self.g0 + to_db(0.5)
        self.hpbw = 2 * min(elev[np.where(self.gain(elev) < target)])

    def directivity(self, elev):
        value = 0
        if self.p_rad is not None:
            value = to_db(self._radiation_pattern(elev) * np.pi * 4 / self.p_rad)
        return value

    def gain(self, elev):
        return self.directivity(elev) + self.losses

    def fspl(self, r):
        lambda_ = SPEED_OF_LIGHT / (self.freq * 1e6)
        return to_db((lambda_ / (4 * np.pi * r)) ** 2)

    def transmission_losses(self, other, distance, angle_self=0, angle_other=0):
        gain_s = self.gain(angle_self)
        gain_o = other.gain(angle_other)
        fspl = self.fspl(distance)

        return gain_s + gain_o + fspl

    def transmit(self, receiver, distance, angle_self=0, angle_other=0):
        return self.transmit_pwr + self.transmission_losses(receiver, distance, angle_self, angle_other)

    def receive(self, transmitter, distance, angle_self=0, angle_other=0):
        return self.sensibility - self.transmission_losses(transmitter, distance, angle_self, angle_other)
