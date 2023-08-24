import numpy as np
from CitySimulator import CityConfigurator
from Utils.Plots import *
from CitySimulator.CityDefines import CityConfigStr


# Define configuration parameters for the city
urban_config = CityConfigStr()
urban_config.map_grid_size = 5 # 3D map discretization settings
urban_config.map_x_len = 600 # Map length along the x-axis [m]
urban_config.map_y_len = 800 # Map length along the y-axis [m]
urban_config.bld_height_avg = 15
urban_config.bld_height_max = 50
urban_config.bld_height_min = 5
urban_config.ave_width = 60 # The width of the avenues (main streets) [m]
urban_config.st_width = 20 # The width of the streets (lanes) [m]
urban_config.blk_size_x = 200 # The width of each block in the city (between avenues)
urban_config.blk_size_y = 200 # The length of each block in the city (between avenues)
urban_config.blk_size_small = 100 # Average small block size (between streets)
urban_config.blk_size_min = 80 # Average min block size (between streets)
urban_config.bld_size_avg = 80
urban_config.bld_size_min = 50
urban_config.bld_dense = 0.001 # The density of the building in each block



# Generate a new city map using the defined configuration
city = CityConfigurator(urban_config=urban_config
, gen_new_map=True # generate a new city based on "urban_config" or load a saved city model
, save_map=True # save the generated city model,
, city_file_name='Data/CityModel/city_map.npy' # the directory to save/load the city model
    )

# Visualize the 3D map of the city
plot_city_3d_map(city)

# Visualize the top view of the city
plot_city_top_view(city, fig_id=1)

plot_show() 
