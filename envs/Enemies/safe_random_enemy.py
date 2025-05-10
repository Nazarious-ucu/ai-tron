import random
from envs.Enteties.position import Position

class SafeRandomEnemy:
    def __init__(self, start_pos: Position, field, wall_value=9):
        self.pos = start_pos
        self.field = field
        self.wall_value = wall_value

    def move(self):
        possible_moves = [0, 1, 2, 3]  # up, right, down, left
        safe_moves = []

        for move in possible_moves:
            new_x, new_y = self.pos.x, self.pos.y
            if move == 0:
                new_y -= 1
            elif move == 1:
                new_x += 1
            elif move == 2:
                new_y += 1
            elif move == 3:
                new_x -= 1

            if 0 <= new_x < self.field.width and 0 <= new_y < self.field.height:
                cell_value = self.field.state[new_y, new_x]
                if cell_value not in [self.wall_value, 4]:
                    safe_moves.append(move)

        if not safe_moves:
            return random.choice(possible_moves)
        else:
            return random.choice(safe_moves)
