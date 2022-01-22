from constants import AX25_T102, AX25_T2, AX25_T103, AX25_P

BIT_STUFFING = 63 / 62
OVERHEAD = 20
BYTE_TO_BIT = 8


def get_frame_time(bitrate, info_size=256, frame_count=7):
    assert frame_count <= 7
    assert info_size <= 256

    cs_time = 256 * (AX25_T102 / 100) / (2 * (1 + AX25_P))
    info_time = frame_count * BIT_STUFFING * BYTE_TO_BIT * (OVERHEAD + info_size) / bitrate
    rr_time = BIT_STUFFING * BYTE_TO_BIT * OVERHEAD / bitrate

    return cs_time + 2 * (AX25_T103 / 100) + info_time + (AX25_T2 / 100) + rr_time


def get_effective_bitrate(bitrate, info_size=256, frame_count=7):
    return BYTE_TO_BIT * frame_count * info_size / get_frame_time(bitrate, info_size, frame_count)
