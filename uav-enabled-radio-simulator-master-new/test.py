import numpy as np 
import matplotlib.pyplot as plt
from Libs.CitySimulator.CityConfigurator import *
from Libs.CitySimulator.CityDefines import *
from Libs.Utils.Plots import *
from Libs.Utils.utils import *
from Libs.ChannelModel.SegmentedChannelModel import *
from Libs.ChannelModel.RadioChannelBase import *
from Libs.ChannelModel.ChannelDefines import *
from Defines import *



# Create a City instance
city_model = CityConfigurator(urban_config=urban_config
, gen_new_map=True # generate a new city based on "urban_config" or load a saved city model
, save_map=True # save the generated city model
, city_file_name='Data/CityMap/city_map.npy' # the directory to save the city model
)

# Define the radio channel model
radio_ch_model = SegmentedChannelModel(ch_param)

# Define user positions ( manually or automatically )
user_poses_auto = city_model.user_scattering(num_user=5, user_altitude=0)

# Generate the radio map
radio_map = radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses_auto, resolution=10)

# Generate 100 random radio maps, each radio map has one UAV trajectory and 5 user positions
num_radio_maps = 100

# Generate user positions for each radio map
user_poses_auto = []

for i in range(num_radio_maps):
    user_poses_auto.append(city_model.user_scattering(num_user=5, user_altitude=0))

# Generate UAV trajectories for each radio map
uav_trajectories = []

for i in range(num_radio_maps):
    uav_trajectory_corners = np.array([
    [100, 100, 60],
    [450, 300, 60],
    [150, 400, 60],
    [500, 500, 60],])

    uav_simple_trajectory = generate_uav_trajectory_from_points(uav_trajectory_corners,
    sampling_resolution= 40)
    uav_trajectories.append(uav_simple_trajectory)

# Generate link stautus for each radio map
link_status = []

for i in range(num_radio_maps):
    link_status.append(city_model.link_status_to_user(q_poses=uav_trajectories[i], user_poses=user_poses_auto[i]))

# Collect the measurement for each radio map, 100 measurements in different radio maps
collected_meas = []

for i in range(num_radio_maps):
    collected_meas.append(radio_ch_model.get_measurement_from_users(city=city_model, q_poses=uav_trajectories[i], user_poses=user_poses_auto[i]))

# Save the radio map separately
for i in range(num_radio_maps):
    np.save('Data/RadioMap/radio_map_'+str(i)+'.npy', radio_map)

# Save the collected measurement separately
for i in range(num_radio_maps):
    np.save('Data/Measurement/collected_meas_'+str(i)+'.npy', collected_meas[i])


#Print the data shape for each of the radio map
print(np.shape(radio_map))

# Print the data shape of the collected measurement
print(np.shape(collected_meas))

#read and print the first and second collected measurement
collected_meas_1 = np.load('Data/Measurement/collected_meas_1.npy', allow_pickle=True)
print(collected_meas_1[0][0].los_status)
print(collected_meas_1[0][0].rssi_db)
print(collected_meas_1[0][0].ch_gain_db)
print(collected_meas_1[0][0].ch_capacity)

collected_meas_2 = np.load('Data/Measurement/collected_meas_2.npy', allow_pickle=True)
print(collected_meas_2[0][0].los_status)
print(collected_meas_2[0][0].rssi_db)
print(collected_meas_2[0][0].ch_gain_db)
print(collected_meas_2[0][0].ch_capacity)
