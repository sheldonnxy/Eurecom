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

# Define user positions ( manually or automatically )
user_poses_auto = city_model.user_scattering(num_user=5, user_altitude=0)
# user_poses_manual = np.array([
# [100, 200, 0],
# [320, 550, 0],
# [550, 450, 0],
# ]) 

#UAV trajectory ( manually or automatically )
uav_trajectory_corners = np.array([
[100, 100, 60],
[450, 300, 60],
[150, 400, 60],
[500, 500, 60],])

uav_simple_trajectory = generate_uav_trajectory_from_points(uav_trajectory_corners,
sampling_resolution= 40)

# Collect measurements
measurements = radio_ch_model.get_measurement_from_users(city=city_model, q_poses=uav_simple_trajectory, user_poses=user_poses_auto, sampling_resolution=10)

# Generate radio map
radio_map = radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses_auto, resolution=10)


# Link status
link_status = city_model.link_status_to_user(q_poses=uav_simple_trajectory, user_poses=user_poses_auto)

# Print link status
print(link_status)

# 3D city map plot
plot_city_3d_map(city_model) # plot the 3D model of the city
plot_city_top_view(city_model, fig_id=1) # plot the top view map of the city

# User plot
plot_user_positions(user_poses_auto, fig_id=1, marker='*', marker_size=130, color='b')

# UAV trajectory plot
plot_uav_trajectory(uav_simple_trajectory, fig_id=1, marker='o', line_width=3.5, color='m', marker_size=8) 


# Measurement plot
plot_rssi_measurements(measurements, fig_id=2, marker='o', color='m')

# Radio map plot
plot_radio_map(radio_map.ch_gain_db, fig_id=2, resolution=1)







# Generate a 0-1 mask for grids around measurment place, the close grids are 1 and other grids are 0


# Concatenate the measurement and mask


# Save the masked collected measurement



# this function should be used at the end of the code
plot_show()