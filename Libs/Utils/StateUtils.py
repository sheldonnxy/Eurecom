import math
import numpy as np
# import sys


def pad_centered(state, map_in, pad_value):
    padding_rows = math.ceil(state.users_map.shape[0] / 2.0)
    padding_cols = math.ceil(state.users_map.shape[1] / 2.0)
    position_x, position_y = state.position
    position_row_offset = padding_rows - position_y
    position_col_offset = padding_cols - position_x
    return np.pad(map_in,
                  pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
                             [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
                             [0, 0]],
                  mode='constant',
                  constant_values=pad_value)
    # try:
    #     return np.pad(map_in,
    #                   pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
    #                              [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
    #                              [0, 0]],
    #                   mode='constant',
    #                   constant_values=pad_value)
    # except:
    #     print("padding_rows", padding_rows, " padding_cols", padding_cols, " position_row_offset", position_row_offset,
    #           " position_col_offset", position_col_offset, " pad_value", pad_value)
    #     print("users_map.shape", state.users_map.shape, " state.position", state.position)
    #     print(sys.exc_info()[0])
    #     raise
