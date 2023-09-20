import numpy as np
import matplotlib.pyplot as plt
from Libs.CitySimulator.CityConfigurator import *
from Libs.CitySimulator.CityDefines import *
from Libs.Utils.Plots import *
from Libs.Utils.utils import *
from Libs.Environments.DataCollection import *
from Libs.ChannelModel.SegmentedChannelModel import *

# City configuration class
urban_config = CityConfigStr()

# Create a City instance
city = CityConfigurator(urban_config=urban_config
, gen_new_map=True # generate a new city based on "urban_config" or load a saved city model
, save_map=True # save the generated city model
, city_file_name='Data/CityMap/city_map.npy' # the directory to save the city model
)



#Define User location ( manually or automatically )
user_poses_auto = city.user_scattering(num_user=3, user_altitude=0) 
user_poses_manual = np.array([
[100, 200, 0],
[320, 550, 0],
[550, 450, 0],
]) 



#UAV trajectory ( manually or automatically )

uav_trajectory_corners = np.array([
[100, 100, 60],
[450, 300, 60],
[150, 400, 60],
[500, 500, 60],])

uav_simple_trajectory = generate_uav_trajectory_from_points(uav_trajectory_corners,
sampling_resolution= 40)



#Link status
link_status = city.link_status_to_user(q_poses=uav_simple_trajectory, user_poses=user_poses_auto)



#Radio channel model
ch_param = ChannelParamStr()
radio_channel = SegmentedChannelModel(ch_param)



#Take the channel measurement from the users
collected_meas = radio_channel.get_measurement_from_users(city=city, q_poses=uav_simple_trajectory, user_poses=user_poses_auto)



#Save the collected measurement
np.save('Data/CityMap/collected_meas.npy', collected_meas, allow_pickle=True)


#Generate a 0-1 mask around measurment
mask = np.zeros((len(uav_simple_trajectory), len(user_poses_auto)))

#Concatenate the measurement and mask
collected_meas = np.concatenate((collected_meas, mask), axis=2)

#Save the masked collected measurement
np.save('Data/CityMap/collected_meas_masked.npy', collected_meas, allow_pickle=True)

#Plot 3D city map
plot_city_3d_map(city) # plot the 3D model of the city
plot_city_top_view(city, fig_id=1) # plot the top view map of the city
plot_show() # this function should be used at the end of the code

#User plot
plot_user_positions(user_poses_manual, fig_id=1, marker='*', marker_size=130, color='b')
plot_user_positions(user_poses_auto, fig_id=1, marker='*', marker_size=130, color='r')
plot_show()

# plot the UAV trajectory over the city map
plot_city_top_view(city, fig_id=1)
plot_uav_trajectory(uav_simple_trajectory, fig_id=1, marker='o', line_width=3.5,
color='m', marker_size=8) 
plot_show() 