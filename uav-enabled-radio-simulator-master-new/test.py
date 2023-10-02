import numpy as np 
import matplotlib.pyplot as plt
from Libs.CitySimulator.CityConfigurator import *
from Libs.CitySimulator.CityDefines import *
from Libs.Utils.Plots import *
from Libs.Utils.utils import *
# from Libs.Environments.DataCollection import *
from Libs.ChannelModel.SegmentedChannelModel import *
from Libs.ChannelModel.RadioChannelBase import *
from Libs.ChannelModel.ChannelDefines import *
from scipy.interpolate import interp2d
from Defines import *



# Create a City instance
city_model = CityConfigurator(urban_config=urban_config
, gen_new_map=True # generate a new city based on "urban_config" or load a saved city model
, save_map=True # save the generated city model
, city_file_name='Data/CityMap/city_map.npy' # the directory to save the city model
)
radio_ch_model = SegmentedChannelModel(ch_param)

user_poses_auto = city_model.user_scattering(num_user=5, user_altitude=0)

radio_map = radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses_auto, resolution=10)

#Plot 3D city map
plot_city_3d_map(city_model) # plot the 3D model of the city
plot_city_top_view(city_model, fig_id=1) # plot the top view map of the city

# #User plot
plot_user_positions(user_poses_auto, fig_id=1, marker='*', marker_size=130, color='b')
# plot_user_positions(user_poses_auto, fig_id=1, marker='*', marker_size=130, color='r')

plot_radio_map(radio_map.ch_gain_db[0], resolution=10) # plot the radio map

# # plot the UAV trajectory over the city map
# plot_city_top_view(city, fig_id=1)
# plot_uav_trajectory(uav_simple_trajectory, fig_id=1, marker='o', line_width=3.5,
# color='m', marker_size=8) 
plot_show() 

# # Create city as grid
# city_grid = np.zeros((city.urban_config.map_x_len, city.urban_config.map_y_len))

# # Randomly select three locations on the city boulevard for the user devices
# user_device_locations = np.array([
#     [np.random.randint(0, city.urban_config.map_x_len), 0],  # Top edge
#     [0, np.random.randint(0, city.urban_config.map_y_len)],  # Left edge
#     [np.random.randint(0, city.urban_config.map_x_len), city.urban_config.map_y_len - 1]  # Bottom edge
# ])


# def generate_radio_measurement_for_user(device_location, channel, grid_shape=(10, 10)):
#     radio_map = np.zeros(grid_shape)
#     for i in range(grid_shape[0]):
#         for j in range(grid_shape[1]):
#             center = np.array([(i + 0.5) * 60, (j + 0.5) * 60])
#             radio_map[i][j] = channel(source_point=device_location, dest_point=center)
#     return radio_map

# all_radio_measurements = []
# for user_location in user_device_locations:
#     measurement = generate_radio_measurement_for_user(user_location, channel)
#     all_radio_measurements.append(measurement)

# plt.imshow(all_radio_measurements[0], cmap='jet', origin='lower')
# plt.colorbar(label='Signal Strength')
# plt.scatter(user_device_locations[0][0] // 60, user_device_locations[0][1] // 60, c='white', s=100, marker='x')
# plt.title('Radio Map for User Device 1')
# plt.show()








# # Visualize the radio map with user devices using a basic 2D heatmap
# # plt.figure(figsize=(10,10))
# # plt.imshow(radio_map, cmap='jet', origin='lower')
# # plt.colorbar(label='Signal Strength')
# # plt.scatter(user_device_locations[:, 0], user_device_locations[:, 1], c='white', s=100, marker='x', label='User Devices')
# # plt.title('Radio Map with User Devices')
# # plt.legend()
# # plt.show()





# #Define User location ( manually or automatically )
# user_poses_auto = city.user_scattering(num_user=3, user_altitude=0) 
# user_poses_manual = np.array([
# [100, 200, 0],
# [320, 550, 0],
# [550, 450, 0],
# ]) 




# #UAV trajectory ( manually or automatically )

# uav_trajectory_corners = np.array([
# [100, 100, 60],
# [450, 300, 60],
# [150, 400, 60],
# [500, 500, 60],])

# uav_simple_trajectory = generate_uav_trajectory_from_points(uav_trajectory_corners,
# sampling_resolution= 40)



# #Link status
# link_status = city.link_status_to_user(q_poses=uav_simple_trajectory, user_poses=user_poses_auto)



# #Radio channel model
# ch_param = ChannelParamStr()
# radio_channel = SegmentedChannelModel(ch_param)



# #Take the channel measurement from the users
# collected_meas = radio_channel.get_measurement_from_users(city=city, q_poses=uav_simple_trajectory, user_poses=user_poses_auto)



# #Save the randomly sampled measurement
# np.save('Data/CityMap/collected_meas.npy', collected_meas, allow_pickle=True)



# #Generate a 0-1 mask around measurment


# #Concatenate the measurement and mask


# #Save the masked collected measurement


# #Plot radio map
# # plot_radio_map(city, radio_map, fig_id=1) # plot the radio map


#Plot 3D city map
# plot_city_3d_map(city) # plot the 3D model of the city
# plot_city_top_view(city, fig_id=1) # plot the top view map of the city
# plot_show() # this function should be used at the end of the code

# #User plot
# plot_user_positions(user_poses_manual, fig_id=1, marker='*', marker_size=130, color='b')
# plot_user_positions(user_poses_auto, fig_id=1, marker='*', marker_size=130, color='r')
# plot_show()

# # plot the UAV trajectory over the city map
# plot_city_top_view(city, fig_id=1)
# plot_uav_trajectory(uav_simple_trajectory, fig_id=1, marker='o', line_width=3.5,
# color='m', marker_size=8) 
# plot_show() 