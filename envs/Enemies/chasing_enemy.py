import random
from dataclasses import field


class ChasingEnemy:
    def __init__(self, start_pos, field):
        self.pos = start_pos
        self.field = field

    def move(self, agent_pos):
        best = None
        best_dist = float('inf')
        for move, (dx,dy) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):
            nx, ny = self.pos.x + dx, self.pos.y + dy
            if not self.field.is_out_of_bounds(nx, ny) and self.field.state[ny][nx] != 4 and self.field.state[ny][nx] != 3 and self.field.state[ny][nx] != 2 and self.field.state[ny][nx] != 1:
                dist = abs(agent_pos.x - nx) + abs(agent_pos.y - ny)
                if dist < best_dist:
                    best_dist, best = dist, move

        # print("Best move: ", best, "Field state: ", self.field.state[ self.pos.y +[(0,-1),(1,0),(0,1),(-1,0)][best][1]][self.pos.x +[(0,-1),(1,0),(0,1),(-1,0)][best][0]])
        # print(self.field.state)
        # print(self.pos.x, self.pos.y)


        if best is not None:
            return best
        else:
            return random.choice([0,1,2,3])
