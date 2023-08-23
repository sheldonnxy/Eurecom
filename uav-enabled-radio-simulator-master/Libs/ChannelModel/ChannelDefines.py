import numpy as np


class ChannelParamStr:
    def __init__(self):
        self.los_exp = 0
        self.los_bias_db = 0
        self.los_var_db = 0
        self.nlos_exp = 0
        self.nlos_bias_db = 0
        self.nlos_var_db = 0
        self.p_tx_db = 0
        self.noise_level_db = 0
        self.band_width = 0


class RadioMeasurementStr:
    def __init__(self):
        self.ch_gain_db = 0
        self.rssi_db = 0
        self.snr_db = 0
        self.ch_capacity = 0
        self.dist = 0
        self.q_pose = np.zeros(shape=[1, 3])
        self.ue_pose = np.zeros(shape=[1, 3])
        self.los_status = 0


class RadioMapStr:
    def __init__(self, shape=None):
        if shape is None:
            shape = [1, 1, 1]
        self.los = np.zeros(shape=shape)
        self.rssi_db = self.los.copy()
        self.ch_gain_db = self.los.copy()
        self.capacity = self.los.copy()


def get_different_ch_output(ch_param=ChannelParamStr(), ch_gain_db=None):
    rssi_db = ch_param.p_tx_db + ch_gain_db

    snr_db = rssi_db - ch_param.noise_level_db
    snr = np.power(10, snr_db / 10)
    capacity = ch_param.band_width * np.log2(1 + snr)
    return ch_gain_db, rssi_db, snr_db, capacity

