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
city_model = CityConfigurator(urban_config=urban_config,
                              gen_new_map=True,
                              save_map=True,
                              city_file_name='Data/CityMap/city_map.npy')

# Define the radio channel model
radio_ch_model = SegmentedChannelModel(ch_param)

# Define UAV trajectory
uav_trajectory_corners = np.array([
    [100, 100, 60],
    [450, 300, 60],
    [150, 400, 60],
    [500, 500, 60],
])

uav_simple_trajectory = generate_uav_trajectory_from_points(
    uav_trajectory_corners,
    sampling_resolution=20
)

# Automatic set user positions for 10 times, then always use the same simple uav trajectory, generate the global radio map and the measurement map accordingly
for i in range(10):
    # Define user positions (automatically)
    user_poses_auto = city_model.user_scattering(num_user=5, user_altitude=0)
    
    # Generate global radio map
    map_resolution = 10
    map_shape = (int(city_model.urban_config.map_y_len // map_resolution),
                 int(city_model.urban_config.map_x_len // map_resolution))
    global_radio_map = np.full(map_shape, -np.inf)
    
    for y in range(map_shape[0]):
        for x in range(map_shape[1]):
            receiver_pose = np.array([x * map_resolution, y * map_resolution, 60])
            rssi_linear_sum = 0
            for user_pose in user_poses_auto:
                link_status = city_model.check_link_status(user_pose, receiver_pose)
                distance = np.linalg.norm(user_pose - receiver_pose)
                _, rssi_db, _, _ = radio_ch_model.channel_response(distance, link_status)
                rssi_linear_sum += 10**(rssi_db / 10)
            global_radio_map[y, x] = 10 * np.log10(rssi_linear_sum)
    
    # Generate measurement map
    measurements = radio_ch_model.get_measurement_from_users(
        city=city_model,
        q_poses=uav_simple_trajectory,
        user_poses=user_poses_auto,
        sampling_resolution=10
    )
    
    measurement_map = np.full(map_shape, np.nan)
    for measurement_list in measurements:
        for measurement in measurement_list:
            x = int(measurement.q_pose[0] // map_resolution)
            y = int(measurement.q_pose[1] // map_resolution)
            if 0 <= x < map_shape[1] and 0 <= y < map_shape[0]:
                measurement_map[y, x] = measurement.rssi_db
    
    # Save the global radio map and the measurement map to the two different directories accordingly
    np.save(f'Data/RadioMap_new/global_radio_map_{i}.npy', global_radio_map)
    np.save(f'Data/MeasurementMap_new/measurement_map_{i}.npy', measurement_map)
    
    # Generate 0-1 mask
    mask_radius = 2
    measurement_mask = np.zeros_like(measurement_map)
    for measurement_list in measurements:
        for measurement in measurement_list:
            x = int(measurement.q_pose[0] // map_resolution)
            y = int(measurement.q_pose[1] // map_resolution)
            if 0 <= x < map_shape[1] and 0 <= y < map_shape[0]:
                x_min, x_max = max(0, x - mask_radius), min(map_shape[1], x + mask_radius + 1)
                y_min, y_max = max(0, y - mask_radius), min(map_shape[0], y + mask_radius + 1)
                measurement_mask[y_min:y_max, x_min:x_max] = 1
    
    # Concatenate each measurement map with the 0-1 mask, then save the masked measurement map to its own directory
    masked_measurement_map = np.stack((measurement_map, measurement_mask), axis=-1)
    np.save(f'Data/MaskedMeasurementMap/masked_measurement_map_{i}.npy', masked_measurement_map)

print("All maps have been generated and saved.")