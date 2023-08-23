from PytorchModel.CityModel import *

#TD: generate a dataset of 3D city models


from Utils.Plots import *
from CitySimulator.CityConfigurator import *

# City configuration class
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



city = CityModel(urban_config=urban_config
                , gen_new_map=True # generate a new city based on "urban_config" or load a saved city model
                , save_map=True # save the generated city model
                , city_file_name='Data/CityMap/city_map.npy' # the directory to save the city model
                )

plot_city_3d_map(city) # plot the 3D model of the city
plot_city_top_view(city, fig_id=1) # plot the top view map of the city
plot_show() # this function should be used at the end of the code


#more details about the city configuration can be found in the CityConfigurator.py file