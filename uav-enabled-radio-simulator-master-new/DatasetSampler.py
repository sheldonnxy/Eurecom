import numpy as np 
import matplotlib.pyplot as plt
from Libs.CitySimulator.CityConfigurator import *
from Libs.CitySimulator.CityDefines import *
from Libs.ChannelModel.SegmentedChannelModel import *
from Libs.ChannelModel.RadioChannelBase import *
from Libs.ChannelModel.ChannelDefines import *
from Libs.Utils.Plots import *
from Libs.Utils.utils import *
from Defines import *



# Create a City instance
city_model = CityConfigurator(urban_config=urban_config
, gen_new_map=True # generate a new city based on "urban_config" or load a saved city model
, save_map=True # save the generated city model
, city_file_name='Data/CityMap/city_map.npy' # the directory to save the city model
)

# Define the radio channel model
radio_ch_model = SegmentedChannelModel(ch_param)

# Define user positions ( automatically )
user_poses_auto = city_model.user_scattering(num_user=5, user_altitude=0)

# Generate 10 random radio maps, each radio map has the same UAV trajectory and 5 user positions
num_radio_maps = 10

radio_maps = []
for i in range(num_radio_maps):
    radio_maps.append(radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses_auto, resolution=10))
    np.save('Data/RadioMap/radio_map_' + str(i) + '.npy', radio_maps[i])

user_poses = []
for i in range(num_radio_maps):
    user_poses.append(user_poses_auto)
    np.save('Data/UserPoses/user_pos_' + str(i) + '.npy', user_poses[i])

# Generate UAV trajectory for each radio map
uav_simple_trajectory = []
uav_trajectory_corners = np.array([
    [100, 100, 60],
    [450, 300, 60],
    [150, 400, 60],
    [500, 500, 60],])
for i in range(num_radio_maps):
    uav_simple_trajectory.append(generate_uav_trajectory_from_points(uav_trajectory_corners,
    sampling_resolution= 40))
    np.save('Data/UAVTrajectory/uav_trajectory_' + str(i) + '.npy', uav_simple_trajectory[i])

# Generate link stautus for each radio map
link_status = []
for i in range(num_radio_maps):
    link_status.append(city_model.link_status_to_user(q_poses=uav_simple_trajectory[i], user_poses=user_poses[i]))
    np.save('Data/LinkStatus/link_status_' + str(i) + '.npy', link_status[i])

# Generate measurement for each radio map
collected_meas = []
for i in range(num_radio_maps):
    collected_meas.append(radio_ch_model.get_measurement_from_users(city=city_model, q_poses=uav_simple_trajectory[i], user_poses=user_poses[i]))
    np.save('Data/Measurement/collected_meas_' + str(i) + '.npy', collected_meas[i])



# Concatanate each measurement a mask



#plot the city map
plot_city_3d_map(city_model) # plot the 3D model of the city
plot_city_top_view(city_model, fig_id=1) # plot the top view map of the city
plot_city_top_view(city_model, fig_id=2) # plot the top view map of the city

# plot the first and second user positions
plot_user_positions(user_poses[0], fig_id=1, marker='*', marker_size=130, color='b')
plot_user_positions(user_poses[1], fig_id=2, marker='*', marker_size=130, color='b')

# plot the first and second UAV trajectory
plot_uav_trajectory(uav_simple_trajectory[0], fig_id=1, marker='o', line_width=3.5, color='m', marker_size=8)
plot_uav_trajectory(uav_simple_trajectory[1], fig_id=2, marker='o', line_width=3.5, color='m', marker_size=8)

# Plot the first and second radio map
radio_map = radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses[0], resolution=10)
plot_radio_map(radio_map.ch_gain_db[0], fig_id=3, resolution=10)
radio_map = radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses[1], resolution=10)
plot_radio_map(radio_map.ch_gain_db[0], fig_id=4, resolution=10)



# Plot the first and second measurement


# Print the mask for the first and second measurement


# this function should be used at the end of the code
plot_show()
