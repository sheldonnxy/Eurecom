from docutils.nodes import citation

from Libs.Utils.utils import *
from Libs.Utils.Plots import *
from Defines import *
import os
from Libs.Controller.ScalarDRLController import *
from tqdm import tqdm
from time import sleep

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----------------> City and Radio Channel <---------------------
city = CityConfigurator(gen_new_map=False, save_map=False, urban_config=urban_config,
                        city_file_name='Data/CityMap/city_map.npy')
radio_channel = SegmentedChannelModel(ch_param)

# -------------------> Collecting measurement from the real world <--------------------
# collected_meas: List of RadioMeasurementStr with size of: [No UAV positions] [No Users]
collected_meas = radio_channel.get_measurement_from_users(city=city, q_poses=uav_trj, user_poses=user_pose,
                                                          sampling_resolution=40)

sampled_uav_trj = np.array([meas[0].q_pose for meas in collected_meas])
all_los_ch_gains = np.array([[ch_gain.ch_gain_db] for meas in collected_meas for ch_gain in meas])
all_meas_dist = np.array([[ch_gain.dist] for meas in collected_meas for ch_gain in meas])

radio_maps = radio_channel.get_radio_map(city, q_height=60, ue_poses=user_pose, resolution=10)

# -------------------> Environment & Controller definition <---------------------------
env_param = EnvParamStr()
env_param.start_pose = uav_start_pose
env_param.end_pos = uav_terminal_pose
env_param.city_model = city
env_param.user_pose = user_pose
env_param.step_size = 20
env_param.battery_budget = 50
env_param.radio_ch_model = radio_channel
env_param.safety_controller_flag = True


env = DataCollection(env_param)  # An IoT scenario in which a UAV moves around to collect data from ground users

hidden_layers = [(150, 'relu'), (150, 'relu')]
controller = ScalarDRLController(hidden_layers, env, epsilon_decay=0.0005)
batch_size = 50
EPISODES = 100
reward_buf = []

print("Starting training...")
sleep(0.1)
pbar = tqdm(total=EPISODES)

for e in range(EPISODES):
    env.reset()
    state = env.agent_state.get_state_vec()
    done = 0
    uav_trj = [env.agent_pose.flatten()]
    while done == 0:
        action = controller.act(state)
        reward, next_state, done, info = env.step(action)

        new_observation = (state, action, reward, next_state, done)
        controller.memorize(new_observation)
        state = next_state
        uav_trj.append(env.agent_pose.flatten())

        if e % 50 == 0:
            controller.update_target_model()
        if e > 1:
            controller.replay(int(min(len(controller.memory), batch_size)))
        if done:
            reward_buf.append(info.collected_data)
            pbar.update(1)
            info = "collected_data: {:.5}, sum_rewards: {:.4}, battery: {:.3}, e: {:.3}, trj_length: {}" \
                .format(info.collected_data, float(tf.reduce_mean(reward_buf[-50:])),
                        state[-1], float(controller.epsilon), len(uav_trj))
            pbar.set_postfix_str(info)
            break

env_uav_trj = np.array([pose for pose in uav_trj])

user_poses_auto = city.user_scattering(num_user=3, user_altitude=0)
# -------------------> Plotting results <--------------------
plot_radio_map(radio_maps.ch_gain_db[0], resolution=10, cb_title='Channel gain [dB]')
plot_city_top_view(city, 10)
plot_user_positions(user_pose, 10, marker='*', marker_size=130, color='b')
# plot_user_positions(user_poses_auto, 10, marker='*', marker_size=130, color='r')
plot_uav_trajectory(sampled_uav_trj, 10, marker='o', line_width=3.5, color='m', marker_size=8)
plot_uav_trajectory(env_uav_trj, 10, marker='^', line_width=4, color='g', marker_size=6)
plot_smooth_reward(range(EPISODES), reward_buf, 5)
plt.figure()
plt.plot(all_meas_dist, all_los_ch_gains, 'b*')
plot_city_3d_map(city)

# To plot in 3D (users, UAV, etc.)
# mlab.points3d(ub_config.map_x_len / 2, ub_config.map_y_len / 2, 60 * 5, 10, color=(0.5, 0.1, 1),
#               scale_factor=ub_config.map_grid_size)

plot_show()
