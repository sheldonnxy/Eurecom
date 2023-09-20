import matplotlib.pyplot as plt
import numpy as np
from numpy import save
from Libs.CitySimulator.CityConfigurator import *
from Libs.ChannelModel.SegmentedChannelModel import *
from Libs.Environments.DataCollection import *

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

# Radio Channel parameters
ch_param = ChannelParamStr()
ch_param.los_exp = -2.5
ch_param.los_bias_db = -30
ch_param.los_var_db = 2
ch_param.nlos_exp = -3.04
ch_param.nlos_bias_db = -35
ch_param.nlos_var_db = 5
ch_param.p_tx_db = 43
ch_param.noise_level_db = -60
ch_param.band_width = 100

uav_height = 60
bs_height = 30

uav_trj = np.array([
    [100, 100, 60],
    [450, 300, 60],
    [150, 400, 60],
    [500, 500, 60],
])
uav_trj[:, -1] = uav_height

# Landmark positions
user_pose = np.array([
    [400, 200, 0],
    [250, 450, 0],
    [350, 400, 0],
    [100, 300, 0],
])

# landmark_pos[:, -1] = uav_height

uav_start_pose = np.array([[100, 100, 60]])
uav_terminal_pose = np.array([[300, 400, 60]])
