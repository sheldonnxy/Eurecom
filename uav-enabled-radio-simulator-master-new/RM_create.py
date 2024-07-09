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
city_model = CityConfigurator(
    urban_config=urban_config,
    gen_new_map=True,  # generate a new city based on "urban_config" or load a saved city model
    save_map=True,  # save the generated city model
    city_file_name='Data/CityMap/city_map.npy'  # the directory to save the city model
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

# Convert to numpy arrays to ensure proper indexing
user_poses_auto = np.array(user_poses_auto)


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

# Collect measurements
measurements = radio_ch_model.get_measurement_from_users(
    city=city_model,
    q_poses=uav_simple_trajectory,
    user_poses=user_poses_auto,
    sampling_resolution=10
)

# Create a measurement map
map_resolution = 10 
map_shape = (int(city_model.urban_config.map_y_len // map_resolution),
             int(city_model.urban_config.map_x_len // map_resolution))
measurement_map = np.full(map_shape, np.nan)
los_status_map = np.full(map_shape, np.nan)

# Fill the measurement map
for measurement_list in measurements:
    for measurement in measurement_list:
        x = int(measurement.q_pose[0] // map_resolution)
        y = int(measurement.q_pose[1] // map_resolution)
        if 0 <= x < map_shape[1] and 0 <= y < map_shape[0]:
            measurement_map[y, x] = measurement.rssi_db


radio_map = radio_ch_model.get_radio_map(city=city_model, q_height=60, ue_poses=user_poses_auto, resolution=10)



# Plot 1: UAV Trajectory
plt.figure(figsize=(12, 10))

plot_city_top_view(city_model, fig_id=plt.gcf().number)
plot_user_positions(user_poses_auto, fig_id=plt.gcf().number, marker='*', marker_size=130, color='b')
plot_uav_trajectory(uav_simple_trajectory, fig_id=plt.gcf().number, marker='o', line_width=3.5, color='m', marker_size=8)

plt.title('City Map with UAV Trajectory')
plt.xlabel('X coordinate (m)')
plt.ylabel('Y coordinate (m)')
plt.legend(['Users', 'UAV Trajectory'])

# Plot 2: RSSI Measurements along UAV Trajectory
plt.figure(figsize=(12, 10))

plot_city_top_view(city_model, fig_id=plt.gcf().number)
plot_user_positions(user_poses_auto, fig_id=plt.gcf().number, marker='*', marker_size=130, color='b')

measurement_mask = ~np.isnan(measurement_map)
im = plt.imshow(np.ma.array(measurement_map, mask=~measurement_mask), 
                extent=[0, city_model.urban_config.map_x_len, 0, city_model.urban_config.map_y_len],
                origin='lower', cmap='jet', alpha=0.7)

plt.colorbar(im, label='RSSI (dB)')
plt.title('RSSI Measurements along UAV Trajectory')
plt.xlabel('X coordinate (m)')
plt.ylabel('Y coordinate (m)')
plt.legend(['Users', 'RSSI Measurements'])



# Plot 3: True Radio Map
plt.figure(figsize=(12, 10))

# Create global radio map
map_resolution = 10
map_shape = (int(city_model.urban_config.map_y_len // map_resolution),
             int(city_model.urban_config.map_x_len // map_resolution))
global_radio_map = np.full(map_shape, -np.inf)
los_map = np.zeros(map_shape, dtype=bool)

for i in range(map_shape[0]):
    for j in range(map_shape[1]):
        x = j * map_resolution
        y = i * map_resolution
        receiver_pose = np.array([x, y, 60])  # Assuming UAV height is 60m
        rssi_linear_sum = 0
        for user_pose in user_poses_auto:
            link_status = city_model.check_link_status(user_pose, receiver_pose)
            distance = np.linalg.norm(user_pose - receiver_pose)
            _, rssi_db, _, _ = radio_ch_model.channel_response(distance, link_status)
            rssi_linear_sum += 10**(rssi_db / 10)
            if link_status == 1:
                los_map[i, j] = True
        global_radio_map[i, j] = 10 * np.log10(rssi_linear_sum)

# Plot the global radio map
im = plt.imshow(global_radio_map, 
                extent=[0, city_model.urban_config.map_x_len, 0, city_model.urban_config.map_y_len],
                origin='lower', cmap='jet')

plt.colorbar(im, label='Combined RSSI (dB)')
plt.title('Global Radio Map (Combined RSSI with NLOS)')
plt.xlabel('X coordinate (m)')
plt.ylabel('Y coordinate (m)')

# Overlay LOS/NLOS areas
plt.imshow(los_map, extent=[0, city_model.urban_config.map_x_len, 0, city_model.urban_config.map_y_len],
           origin='lower', cmap='gray', alpha=0.3)



# Plot 4: Measurement Mask Map
plt.figure(figsize=(12, 10))

# Generate a 0-1 mask for grids on the measurement point, the close grids are 1 and other grids are 0
mask_radius = 2  # Number of cells around each measurement point to consider
measurement_mask = np.zeros_like(global_radio_map)

for measurement_list in measurements:
    for measurement in measurement_list:
        x = int(measurement.q_pose[0] // map_resolution)
        y = int(measurement.q_pose[1] // map_resolution)
        if 0 <= x < map_shape[1] and 0 <= y < map_shape[0]:
            x_min, x_max = max(0, x - mask_radius), min(map_shape[1], x + mask_radius + 1)
            y_min, y_max = max(0, y - mask_radius), min(map_shape[0], y + mask_radius + 1)
            measurement_mask[y_min:y_max, x_min:x_max] = 1

plt.imshow(measurement_mask, extent=[0, city_model.urban_config.map_x_len, 0, city_model.urban_config.map_y_len],
           origin='lower', cmap='binary', alpha=0.7)

plt.title('Measurement Mask Map')
plt.xlabel('X coordinate (m)')
plt.ylabel('Y coordinate (m)')



# Plot 5: Masked Measurement Map
plt.figure(figsize=(12, 10))

# Apply the measurement mask to the global radio map
masked_radio_map = np.where(measurement_mask == 1, global_radio_map, np.nan)

# Plot the masked radio map
im = plt.imshow(masked_radio_map, 
                extent=[0, city_model.urban_config.map_x_len, 0, city_model.urban_config.map_y_len],
                origin='lower', cmap='jet')

plt.colorbar(im, label='Combined RSSI (dB)')
plt.title('Masked Radio Map (Combined RSSI with NLOS)')
plt.xlabel('X coordinate (m)')
plt.ylabel('Y coordinate (m)')

# Overlay LOS/NLOS areas
masked_los_map = np.where(measurement_mask == 1, los_map, np.nan)
plt.imshow(masked_los_map, extent=[0, city_model.urban_config.map_x_len, 0, city_model.urban_config.map_y_len],
           origin='lower', cmap='gray', alpha=0.3)

# Save the masked radio map
np.save('Data/RadioMap/masked_radio_map.npy', masked_radio_map)



# This function should be used at the end of the code
plot_show()