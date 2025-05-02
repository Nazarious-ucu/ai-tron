import random

class RandomEnemy:
    def __init__(self, start_pos):
        self.pos = start_pos

    def move(self):
        return random.choice([0, 1, 2, 3])