from Libs.ChannelModel.RadioChannelBase import *


class LoSChannelModel(RadioChannelBase):
    def __init__(self, ch_param=ChannelParamStr()):
        super().__init__()
        self.ch_param = ch_param

    def collect_measurements(self, city, q_poses, user_poses=None):
        num_users = user_poses.shape[0]
        num_q = q_poses.shape[0]

        user_status = np.ones(shape=[num_q, num_users])
        measurements = []

        for i_q in range(num_q):
            q_pose = q_poses[i_q]
            users_meas = []
            for i_ue in range(num_users):
                meas = RadioMeasurementStr()
                meas.dist = np.linalg.norm((user_poses[i_ue] - q_pose))
                meas.ch_gain_db, meas.rssi_db, meas.snr_db, meas.ch_capacity = self.channel_response(meas.dist, user_status[i_q, i_ue])
                meas.los_status = user_status[i_q, i_ue]
                meas.q_pose = q_pose
                meas.ue_pose = user_poses[i_ue]

                users_meas.append(meas)
            measurements.append(users_meas)

        return measurements

    def channel_response(self, dist, link_status):
        log_dist = np.log10(dist)
        ch_gain_los = self.ch_param.los_bias + self.ch_param.los_exp * log_dist + np.random.normal(
            scale=self.ch_param.los_var)
        ch_gain_db = ch_gain_los

        return get_different_ch_output(self.ch_param, ch_gain_db)

