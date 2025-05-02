import numpy as np


class FieldManager:
    def __init__(self, width, height, wall_value=-1):
        self.width = width
        self.height = height
        self.wall_value = wall_value
        self.state = np.zeros((self.height, self.width), dtype=int)

    def reset_field(self):
        self.state = np.zeros((self.height, self.width), dtype=int)
        # self.state[0, :] = self.wall_value
        # self.state[:, 0] = self.wall_value
        # self.state[-1, :] = self.wall_value
        # self.state[:, -1] = self.wall_value

    def is_out_of_bounds(self, x, y):
        return x >= self.width or y >= self.height or x < 0 or y < 0

    def update_cell(self, x, y, value):
        self.state[y, x] = value