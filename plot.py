import matplotlib as mpl
from matplotlib import pyplot as plt

# Graph Options
OUTPUT_PATH = r'../images/'
USE_PGF = True
PAGE_WIDTH = 6.296
SIZE_FULLPAGE = (PAGE_WIDTH, 9)
SIZE_TALL = (PAGE_WIDTH, 5)
SIZE_DEFAULT = (PAGE_WIDTH, 3)
SIZE_SHORT = (PAGE_WIDTH, 1.25)
X_RESOLUTION = 1024
N_LINES = 4

# Ranges and variables
MIN_ALTITUDE = 200  # km
MAX_ALTITUDE = 6_000  # km
GSO_ALTITUDE = 50_000  # km
FREQUENCIES = (145, 437)  # MHz
ELEV = (10, 15, 30, 45, 60)  # degrees
MIN_ELEV = min(ELEV)
LAT_AND_INC = tuple(range(0, 91, 15))
REF_ALT = MAX_ALTITUDE / 2
REF_ECC = 0.05
NUM_DAYS = 730
MAX_TRACKING = 6
MAX_TIME = 48 * 60 * 60
MAX_SCALED_AXIS = 3
MAX_DEVIATION = 20


def scale_time(interval):
    scales = {
        's': 1,
        'min': 60,
        'h': 3600,
        'd': 86400,
    }
    result = (1, 's')
    for label, scale in scales.items():
        if interval <= scale * MAX_SCALED_AXIS:
            break
        result = (scale, label)
    return result


def scale_data(interval):
    scales = {
        'b': 1,
        'B': 8,
        'KiB': 8192,
        'MiB': 8388608,
    }
    result = (1, 'b')
    for label, scale in scales.items():
        if interval <= scale * MAX_SCALED_AXIS:
            break
        result = (scale, label)
    return result


class Graph(object):
    if USE_PGF:
        mpl.use('pgf')
        mpl.rcParams.update({
            'text.usetex': True,
        })
    plt.style.use('seaborn')

    def __init__(self, filepath, graph_size=SIZE_DEFAULT, legend=False, tight=True, verbose=True):
        self.path = OUTPUT_PATH + filepath + ('.pgf' if USE_PGF else '.png')
        self.size = graph_size
        self.legend = legend
        self.tight = tight
        self.fig = None
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(f'Plotting {self.path}')
        plt.clf()
        self.fig = plt.gcf()
        self.fig.set_size_inches(*self.size)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tight:
            self.fig.tight_layout()
        if self.legend:
            self.fig.legend()
        self.fig.savefig(self.path)
        plt.clf()
        if self.verbose:
            print(f'Finished plotting {self.path}')
        if USE_PGF:
            with open(self.path, 'r') as f:
                data = f.read()
            with open(self.path, 'w') as f:
                f.write(data.replace('°', r'\degree'))


if __name__ == '__main__':
    import numpy as np
    from constants import *
    from contact import Antenna, Contact, GroundStation, Spacecraft, get_angle
    from link_distance import get_link_distance, get_link_aperture_sat
    from timed_contact import align, get_max_comm_window, get_max_comm_window_numeric, get_max_equatorial_comm_window
    from ax25_times import get_effective_bitrate


    def sc_pattern(theta):
        return (
                0.035395870751032045 +
                0.0812061856472038 * np.cos(theta) ** 6
                + np.cos(theta / 2) ** 8.923841918304257
        )


    def gs_pattern(theta):
        theta_array = np.where(np.abs(theta) > np.pi / 2, np.pi / 2, theta)
        return np.cos(theta_array) ** 14.811388300841896


    # Radio configuration
    radio_bitrate = 2400

    sc_radio = {
        'freq': FREQUENCIES[-1],
        'vswr': 1.20,
        'radiation_efficiency': 0.6583423465402455,
        'radiation_pattern': sc_pattern,
        'transmit_pwr': 30,
        'sensibility': -120
    }

    gs_radio = {
        'freq': FREQUENCIES[-1],
        'vswr': 1.20,
        'radiation_efficiency': 1,
        'radiation_pattern': gs_pattern,
        'transmit_pwr': 30,
        'sensibility': -120
    }

    sc_antenna = Antenna(**sc_radio)
    gs_antenna = Antenna(**gs_radio)

    # Orbit configuration
    iss = Spacecraft(
        sma=6798.5e3,
        inc=51.6430,
        aop=116.7490,
        raan=96.2603,
        ecc=0.0004024,
        f0=26.1567
    )
    iss.set_antenna(sc_antenna)

    molnya = Spacecraft(
        sma=((BODY_PERIOD / (4 * np.pi)) ** 2 * BODY_MU) ** (1 / 3),
        inc=np.degrees(np.arcsin(np.sqrt(4 / 5))),
        aop=270,
        raan=-90,
        ecc=0.737,
        f0=180
    )
    molnya.set_antenna(sc_antenna)

    sso_n_orbits = 15
    sso_rot_period = 1.99096871e-7  # 2 * pi radians per sidereal year
    sso_n = sso_n_orbits * 2 * np.pi / BODY_PERIOD
    sso_sma = (sso_n ** -2 * BODY_MU) ** (1 / 3)
    sso_inc = np.arccos(-2 * sso_rot_period / (3 * sso_n * BODY_J2 * (BODY_RADIUS / sso_sma) ** 2))

    sso = Spacecraft(
        sma=sso_sma,
        inc=np.degrees(sso_inc),
        aop=0,
        raan=0,
        ecc=0,
        f0=0
    )
    sso.set_antenna(sc_antenna)

    polar = Spacecraft(
        sma=sso_sma,
        inc=90,
        aop=0,
        raan=0,
        ecc=0,
        f0=0
    )
    polar.set_antenna(sc_antenna)

    # Ground Stations
    ufmg = GroundStation(-19.870682, -43.9699246, MIN_ELEV)
    ufmg.set_antenna(gs_antenna)

    artic = GroundStation(66.5, 0, MIN_ELEV)
    artic.set_antenna(gs_antenna)

    north = GroundStation(90, 0, MIN_ELEV)
    north.set_antenna(gs_antenna)

    null_island = GroundStation(0, 0, MIN_ELEV)

    # Common variables
    altitude = np.linspace(MIN_ALTITUDE, MAX_ALTITUDE, X_RESOLUTION)
    radius = altitude * 1e3 + BODY_RADIUS  # meters

    altitude_gso = np.linspace(MIN_ALTITUDE, GSO_ALTITUDE, X_RESOLUTION)
    radius_gso = altitude_gso * 1e3 + BODY_RADIUS  # meters

    # Gain
    with Graph('gain_antenna', legend=True, graph_size=(0.4 * PAGE_WIDTH, SIZE_DEFAULT[1])) as g:
        elev = np.linspace(-np.pi, np.pi, X_RESOLUTION)
        ax = g.fig.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.plot(elev, sc_antenna.gain(elev), label='Transmissora')
        ax.plot(elev, gs_antenna.gain(elev), label='Receptora')
        ax.set_ylim([-15, 20])

    # View angle
    with Graph('view_angle'):
        for h in ELEV:
            plt.plot(altitude, np.degrees(get_link_aperture_sat(radius, np.radians(h))), label=r'$h = %s°$' % h)

        plt.ylabel('Ângulo de visada ($°$)')
        plt.xlabel('Altitude do apogeu ($km$)')
        plt.gca().legend()

    # Link budget
    with Graph('free_space_path_loss', graph_size=SIZE_TALL) as g:
        axs = g.fig.subplots(nrows=len(FREQUENCIES), sharex=True)

        for ax, f in zip(axs, FREQUENCIES):
            ax.set_ylabel('FSPL ($db$)')
            ax.set_title(f'{f} MHz')
            for h in ELEV:
                antenna = Antenna(f)
                fspl = antenna.fspl(get_link_distance(radius, np.radians(h)))
                ax.plot(altitude, fspl, label=f'$h = %s°$' % h)

        plt.xlabel('Altitude do apogeu ($km$)')
        axs[0].legend()

    # Minimum transmit power
    with Graph('min_transmit_power'):
        sc_radio_lossless = sc_radio.copy()
        sc_radio_lossless['vswr'] = 1
        sc_antenna_lossless = Antenna(**sc_radio_lossless)

        gs_radio_lossless = gs_radio.copy()
        gs_radio_lossless['vswr'] = 1
        gs_antenna_lossless = Antenna(**gs_radio_lossless)

        elevation = get_link_aperture_sat(radius, np.radians(MIN_ELEV))
        distance = get_link_distance(radius, np.radians(MIN_ELEV))
        transmit_pwr_lossy = gs_antenna.sensibility - (
                sc_antenna.gain(elevation) +
                gs_antenna.gain(0) +
                sc_antenna.fspl(distance)
        )

        transmit_pwr = gs_antenna_lossless.sensibility - (
                sc_antenna_lossless.gain(0) +
                gs_antenna_lossless.gain(0) +
                sc_antenna_lossless.fspl(radius - BODY_RADIUS)
        )

        plt.fill_between(altitude, transmit_pwr, transmit_pwr_lossy, alpha=0.25)
        plt.plot(altitude, transmit_pwr)
        plt.ylabel('Potência mínima ($dBm$)')
        plt.xlabel('Altitude do apogeu ($km$)')

    # Equatorial comm window
    with Graph('comm_window'):
        time_scale, time_label = scale_time(MAX_TIME)
        times = np.array([
            get_max_comm_window(sma, 0, 0, np.radians(MIN_ELEV)) for sma in radius_gso
        ])

        gso_time = max(times)
        gso_altitude = altitude_gso[np.where(times == gso_time)][0]
        times[np.where(times > MAX_TIME)] = MAX_TIME * 1.1

        plt.plot(altitude_gso, times / time_scale)
        plt.plot((gso_altitude, gso_altitude), (0, MAX_TIME / time_scale), '--', color='C2')
        plt.ylabel(f'Duração da janela de comunicação (${time_label}$)')
        plt.xlabel('Altitude do apogeu ($km$)')
        plt.ylim(top=MAX_TIME / time_scale)

    # Influence of eccentricity
    with Graph('comm_window_ecc', graph_size=SIZE_TALL) as g:
        axs = g.fig.subplots(nrows=2)

        times_ecc = np.array([
            [
                get_max_comm_window_numeric(null_island, Spacecraft(sma, REF_ECC, 0, 0, aop, 0))
                for sma in radius
            ]
            for aop in (0, 180)
        ])

        time_scale, time_label = scale_time(np.max(times_ecc) - np.min(times_ecc))
        times = get_max_equatorial_comm_window(radius, np.radians(MIN_ELEV)) / time_scale
        times_ecc = times_ecc / time_scale

        axs[0].plot(altitude, times)
        axs[0].fill_between(altitude, times_ecc[0], times_ecc[1], alpha=0.25)
        axs[0].set_xlabel('Altitude do apogeu ($km$)')
        axs[0].set_ylabel(f'Duração da janela\nde comunicação (${time_label}$)')

        sma = REF_ALT * 1e3 + BODY_RADIUS
        ref_alt_time = get_max_equatorial_comm_window(sma, np.radians(MIN_ELEV)) / time_scale
        ref_alt_delta = 2 * (np.max(np.abs([np.interp(REF_ALT, altitude, t) for t in times_ecc])) - ref_alt_time)
        axs[0].plot(
            (REF_ALT, REF_ALT),
            (ref_alt_time + ref_alt_delta, ref_alt_time - ref_alt_delta),
            '--', color='C2'
        )

        time_ref = get_max_equatorial_comm_window(sma, np.radians(MIN_ELEV))
        ecc_axis = np.linspace(0, REF_ECC, X_RESOLUTION)
        times_ecc = np.array([
            [
                get_max_comm_window_numeric(null_island, Spacecraft(sma, ecc, 0, 0, aop, 0))
                for ecc in ecc_axis
            ]
            for aop in (0, 180)
        ]) - time_ref
        time_scale, time_label = scale_time(np.max(times_ecc) - np.min(times_ecc))
        times_ecc = times_ecc / time_scale

        axs[1].plot(ecc_axis, times_ecc[0], color='C2')
        axs[1].plot(ecc_axis, times_ecc[1], color='C2')
        axs[1].fill_between(ecc_axis, times_ecc[0], times_ecc[1], alpha=0.25, color='C2')
        axs[1].set_xlabel('Excentricidade')
        axs[1].set_ylabel(f'Variação na duração da janela\nde comunicação (${time_label}$)')

    # Influence of inclination and latitude
    with Graph('comm_window_lat', graph_size=SIZE_TALL) as g:
        axs = g.fig.subplots(nrows=2)

        times_inc = np.array([
            get_max_comm_window(sma, 0, np.pi / 2, np.radians(MIN_ELEV))
            for sma in radius
        ])

        time_scale, time_label = scale_time(np.max(times_inc) - np.min(times_inc))
        times = get_max_equatorial_comm_window(radius, np.radians(MIN_ELEV)) / time_scale
        times_inc = times_inc / time_scale

        axs[0].plot(altitude, times)
        axs[0].fill_between(altitude, times, times_inc, alpha=0.25)
        axs[0].set_xlabel('Altitude do apogeu ($km$)')
        axs[0].set_ylabel(f'Duração da janela\nde comunicação (${time_label}$)')

        sma = REF_ALT * 1e3 + BODY_RADIUS
        ref_alt_time = get_max_equatorial_comm_window(sma, np.radians(MIN_ELEV)) / time_scale
        ref_alt_delta = 2 * (np.interp(REF_ALT, altitude, times_inc) - ref_alt_time)
        axs[0].plot(
            (REF_ALT, REF_ALT),
            (ref_alt_time + ref_alt_delta, ref_alt_time - ref_alt_delta),
            '--', color='C2'
        )

        time_ref = get_max_equatorial_comm_window(sma, np.radians(MIN_ELEV))
        angle = np.linspace(0, 90, X_RESOLUTION)
        times_max = np.array([
            get_max_comm_window(sma, i, i, np.radians(MIN_ELEV))
            for i in np.radians(angle)
        ]) - time_ref
        times = np.array([
            get_max_comm_window(sma, 0, i, np.radians(MIN_ELEV))
            for i in np.radians(angle)
        ]) - time_ref

        time_scale, time_label = scale_time(np.max(times_max) - np.min(times_max))
        axs[1].plot(angle, times / time_scale, color='C2')
        axs[1].fill_between(angle, times / time_scale, times_max / time_scale, alpha=0.25, color='C2')
        axs[1].set_xlabel('Inclinação ($°$)')
        axs[1].set_ylabel(f'Variação na duração da janela\nde comunicação (${time_label}$)')
        axs[1].set_xticks(range(0, 91, 15))

    # Contact histogram
    with Graph('comm_window_hist'):
        max_time = get_max_comm_window_numeric(ufmg, iss)
        time_scale, time_label = scale_time(max_time)
        times = np.linspace(0, max(max_time - 1, 0), X_RESOLUTION) / time_scale
        max_time_scaled = max_time / time_scale
        prob = times / (max_time_scaled ** 2 * np.sqrt(1 - (times / max_time_scaled) ** 2))

        num_orbits = NUM_DAYS * BODY_PERIOD / iss.period
        u = np.linspace(0, 2 * np.pi * num_orbits, 2 + int(num_orbits * iss.period))
        contacts = Contact.get_contacts(u, ufmg, iss)

        plt.hist([contact.duration / time_scale for contact in contacts], density=True, bins=16)
        ylim = plt.ylim()
        plt.plot(times, prob, '--', color='C2')
        plt.ylim(ylim)

        plt.ylabel(f'Densidade de probabilidade (${time_label}^{{-1}}$)')
        plt.xlabel(f'Duração da janela de comunicação (${time_label}$)')

    # Max data transfer
    with Graph('data_hist'):
        effective_bitrate = get_effective_bitrate(radio_bitrate)
        max_data = max_time * effective_bitrate
        data_scale, data_label = scale_data(max_data)
        datarate = np.linspace(0, max(max_data - 1, 0), X_RESOLUTION) / data_scale
        max_data_scaled = max_data / data_scale
        prob = datarate / (max_data_scaled ** 2 * np.sqrt(1 - (datarate / max_data_scaled) ** 2))

        plt.hist([effective_bitrate * contact.duration / data_scale for contact in contacts], density=True, bins=16)
        ylim = plt.ylim()
        plt.plot(datarate, prob, '--', color='C2')
        plt.ylim(ylim)

        plt.ylabel(f'Densidade de probabilidade (${data_label}^{{-1}}$)')
        plt.xlabel(f'Payload transmitida na janela de comunicação (${data_label}$)')

    # Orbit
    with Graph('orbit', graph_size=SIZE_FULLPAGE) as g:
        labels = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
        orbits = ('ISS', 'Héliossíncrona', 'Polar', 'Molnya')
        pairs = ((iss, ufmg), (sso, ufmg), (polar, north), (molnya, artic))

        contacts = list()
        contacts_alt = list()

        for sc, gs in pairs:
            u = np.linspace(-np.pi, np.pi, 2 + int(sc.period))
            result = list()
            result_alt = list()
            gs_mod, sc_mod = align(gs, sc)
            result.append(Contact.get_contacts(u, gs_mod, sc_mod)[0])

            for lon in range(1, 91):
                if not gs_mod.as_copy(lon=lon).has_sight(sc_mod, 0)[2]:
                    break
            aperture = lon / N_LINES

            for i in range(1, N_LINES):
                result.extend(Contact.get_contacts(
                    u, gs_mod.as_copy(lon=i * aperture), sc_mod
                )[:1])

                result_alt.extend(Contact.get_contacts(
                    u, gs_mod.as_copy(lon=-i * aperture), sc_mod
                )[:1])

            contacts.append(result)
            contacts_alt.append(result_alt)

        n_graphs = 4
        main_gs = plt.GridSpec(n_graphs, 1, figure=g.fig)
        weights = (1, 2)

        for i in range(n_graphs):
            side = bool(i % 2)
            set_gs = main_gs[i].subgridspec(1, 2, width_ratios=weights[::-1 if side else 1], wspace=0.15)

            ax_h = g.fig.add_subplot(set_gs[side], polar=True)
            ax_h.set_theta_zero_location('N')
            ax_h.set_ylim([90, 0])
            ax_h.set_xticks(ax_h.get_xticks(), labels)
            ax_h.set_yticks([30, 60])

            inner_gs = set_gs[not side].subgridspec(2, 1, hspace=0.1)

            ax_p = g.fig.add_subplot(inner_gs[0])
            ax_t = g.fig.add_subplot(inner_gs[1])
            ax_p.set(xticklabels=[])
            ax_p.tick_params(left=False)
            if not side:
                ax_p.yaxis.tick_right()
                ax_p.yaxis.set_label_position('right')
                ax_t.yaxis.tick_right()
                ax_t.yaxis.set_label_position('right')
            else:
                ax_h.yaxis.set_label_position('right')

            time_scale, time_label = scale_time(max(
                [contact.sc.time[-1] - contact.sc.time[0] for contact in contacts[i]]
            ))

            ax_p.set_ylabel(r'$P_r$ ($dBm$)')
            ax_t.set_ylabel(r'$|{{\dot{\vec{\phi}}}_r}|$ ($° \slash s$)')
            ax_t.set_xlabel(f'Tempo $({time_label})$')

            ax_h.set_ylabel(orbits[i], labelpad=40, fontweight='bold')

            max_tracking = list()
            tracking_limit = False
            for j, contact in enumerate(contacts[i]):
                ax_h.plot(contact.azimuth, np.degrees(contact.elevation), color=f'C{j}')
                if j > 0:
                    contact_alt = contacts_alt[i][j - 1]
                    ax_h.plot(contact_alt.azimuth, np.degrees(contact_alt.elevation), color=f'C{j}')
                t0 = min(contact.sc.time[np.where(contact.elevation == max(contact.elevation))])
                sc_time = (contact.sc.time - t0) / time_scale
                ax_p.plot(sc_time, contact.power)
                tracking = np.degrees(np.linalg.norm(contact.tracking, axis=0))
                ax_t.plot(sc_time, tracking)
                max_tracking.append(max(tracking[np.where(tracking < MAX_TRACKING)]))
                if max(tracking) > MAX_TRACKING:
                    tracking_limit = True
            if tracking_limit:
                ax_t.set_ylim([0, 1.25 * max(max_tracking)])

    with Graph('gain_sensibility'):
        max_view_angle = get_link_aperture_sat(radius[0], np.radians(MIN_ELEV))
        theta = np.linspace(0, max_view_angle, X_RESOLUTION // 4).reshape((X_RESOLUTION // 4, 1))
        delta_theta = np.linspace(0, np.radians(MAX_DEVIATION), X_RESOLUTION)
        gains = sc_antenna.gain(theta + delta_theta.reshape((1, X_RESOLUTION))) - sc_antenna.gain(theta)
        plt.fill_between(np.degrees(delta_theta), np.max(gains, axis=0), np.min(gains, axis=0), alpha=0.25)
        plt.plot(np.degrees(delta_theta), sc_antenna.gain(delta_theta) - sc_antenna.gain(0), label='Transmissora')
        plt.plot(np.degrees(delta_theta), gs_antenna.gain(delta_theta) - gs_antenna.gain(0), label='Receptora')
        plt.ylabel('Variação no ganho ($dBm$)')
        plt.xlabel('Erro no apontamento ($°$)')
        plt.gca().legend()

    with Graph('tracking_sensibility', graph_size=SIZE_TALL) as g:
        contact = contacts[0][0]
        axs = g.fig.subplots(nrows=3, sharex=True)

        time_scale, time_label = scale_time(contact.sc.time[-1] - contact.sc.time[0])
        sc_time = contact.sc.time / time_scale

        tracked_elev = contact.elevation
        az_avg = (contact.azimuth[-1] + contact.azimuth[0]) / 2
        slope = np.sign(contact.azimuth[-1]) * contact.sc.time * np.radians(MAX_TRACKING)
        tracked_az = np.where(np.abs(contact.azimuth - az_avg) < np.abs(slope), contact.azimuth, slope + az_avg)

        axs[0].plot(sc_time, np.degrees(contact.elevation), label='Satélite')
        axs[0].plot(sc_time, np.degrees(tracked_elev), '--', color='C2', label='Rastreio')
        axs[0].set_ylabel(r'Elevação ($°$)')
        axs[0].legend()

        axs[1].plot(sc_time, np.degrees(contact.azimuth))
        axs[1].plot(sc_time, np.degrees(tracked_az), '--', color='C2')
        axs[1].set_ylabel(r'Azimute ($°$)')

        sc_pos = np.array([
            np.cos(contact.azimuth) * np.cos(contact.elevation),
            np.sin(contact.azimuth) * np.cos(contact.elevation),
            np.sin(contact.elevation)
        ])

        tracked_pos = np.array([
            np.cos(tracked_az) * np.cos(tracked_elev),
            np.sin(tracked_az) * np.cos(tracked_elev),
            np.sin(tracked_elev)
        ])

        angles = get_angle(sc_pos, tracked_pos)
        angles[np.where(np.isnan(angles))] = 0
        axs[2].plot(sc_time, np.degrees(angles))
        axs[2].set_ylabel(r'Erro ($°$)')
        axs[2].set_xlabel(f'Tempo $({time_label})$')
