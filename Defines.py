import matplotlib.pyplot as plt
import numpy as np
from numpy import save
from Libs.CitySimulator.CityConfigurator import *
from Libs.ChannelModel.SegmentedChannelModel import *
# from Libs.Environments.DataCollection import *
from Libs.Environments.IoTDevice import IoTDevice

# City topology
urban_config = CityConfigStr()

urban_config.map_grid_size = 5  # 3D map discretization settings
urban_config.map_x_len = 600  # Map length along the x-axis [m]
urban_config.map_y_len = 800  # Map length along the y-axis [m]
urban_config.ave_width = 60  # The width of the avenues (main streets) [m]
urban_config.st_width = 20  # The width of the streets (lanes) [m]
urban_config.blk_size_x = 200  # The width of each block in the city (between avenues)
urban_config.blk_size_y = 200  # The length of each block in the city (between avenues)
urban_config.blk_size_small = 100  # Average small block size (between streets)
urban_config.blk_size_min = 80  # Average min block size (between streets)
urban_config.bld_height_avg = 15
urban_config.bld_height_max = 50
urban_config.bld_height_min = 5
urban_config.bld_size_avg = 80
urban_config.bld_size_min = 50
urban_config.bld_dense = 0.001  # The density of the building in each block

city = CityConfigurator(gen_new_map=False, save_map=False, urban_config=urban_config,
                        city_file_name='Data/CityMap/city_map.npy')

# Radio Channel parameters
ch_param = ChannelParamStr()
ch_param.los_exp = -2.5
ch_param.los_bias_db = -30
ch_param.los_var_db = np.sqrt(2)
ch_param.nlos_exp = -3.04
ch_param.nlos_bias_db = -35
ch_param.nlos_var_db = np.sqrt(5)
ch_param.p_tx_db = 43
ch_param.noise_level_db = -60
ch_param.band_width = 100

radio_ch_model = SegmentedChannelModel(ch_param)

uav_height = 60
bs_height = 30

ColorMap = ["brown", "orange", "green", "red", "purple", "blue", "pink", "gray", "olive", "cyan", "black"]

# Landmark positions
# commu1
# known_user_positions = np.array([
#     [[320, 360, 0]],
#     [[120, 460, 0]],
#     [[520, 460, 0]]])

# commu3
device_position = np.array([
    [[60, 320, 0]],
    [[60, 720, 0]],
    [[180, 200, 0]],
    [[240, 460, 0]],
    [[320, 120, 0]],
    [[320, 560, 0]],
    [[320, 760, 0]],
    [[520, 200, 0]],
    [[440, 460, 0]],
    [[460, 720, 0]]])
# datas = np.array([13000, 13000, 13000, 5000, 13000, 13000, 13000, 13000, 16000])
uav_start_pose = np.array([
    [[60, 120, 65]],
    [[60, 120, 60]],
    [[60, 120, 55]]])
uav_terminal_pose = np.array([
    [[560, 720, 65]],
    [[560, 720, 60]],
    [[560, 720, 55]]])
# uav_battery_budget = np.array([60.0, 60.0])
# device_position = np.array([
#     [[20, 200, 0]],
#     [[60, 320, 0]],
#     [[60, 760, 0]],
#     [[120, 600, 0]],
#     [[200, 80, 0]],
#     [[320, 120, 0]],
#     [[320, 760, 0]],
#     [[460, 720, 0]],
#     [[460, 200, 0]],
#     [[580, 520, 0]]
# ])
known_user_idx = np.array([1, 3, 7])
unknown_user_idx = np.array([0, 2, 4, 5, 6, 8, 9])
datas = np.array([16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000])

# uav_start_pose = np.array([
#     [[320, 460, 55]],
#     [[320, 460, 60]],
#     [[320, 460, 65]]])
# uav_terminal_pose = np.array([
#     [[320, 460, 55]],
#     [[320, 460, 60]],
#     [[320, 460, 65]]])
uav_battery_budget = np.array([75.0, 75.0, 75.0])

# uav_start_pose = np.array([
#     [[300, 440, 60]],
#     [[320, 440, 60]],
#     [[340, 440, 60]],
#     [[300, 460, 60]],
#     [[320, 460, 60]],
#     [[340, 460, 60]],
#     [[300, 480, 60]],
#     [[320, 480, 60]],
#     [[340, 480, 60]]
# ])
# uav_terminal_pose = np.array([
#     [[300, 440, 60]],
#     [[320, 440, 60]],
#     [[340, 440, 60]],
#     [[300, 460, 60]],
#     [[320, 460, 60]],
#     [[340, 460, 60]],
#     [[300, 480, 60]],
#     [[320, 480, 60]],
#     [[340, 480, 60]]
# ])
# uav_height = [55, 57, 59, 61, 63, 65]
# uav_battery_budget = np.array([60.0])

colors = ColorMap[:len(datas)]
devices_params = {'position': device_position, 'color': colors, 'data': datas, 'user_num': len(device_position),
                  'known_user_idx': known_user_idx, 'unknown_user_idx': unknown_user_idx}
agent_params = {'start_pose': uav_start_pose, 'end_pose': uav_terminal_pose,
                'battery_budget': uav_battery_budget}
