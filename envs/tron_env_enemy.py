import numpy as np
import pygame
from envs import TronBaseEnv
from envs.Enemies.safe_random_enemy import SafeRandomEnemy
from envs.Enteties.position import Position
import math

def point_in_poly(x, y, poly):
    # простий ray-casting алгоритм
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1)%n]
        # перевірка, чи перетинає горизонтальна лінія
        if ((y0 > y) != (y1 > y)) and (x < (x1-x0)*(y-y0)/(y1-y0) + x0):
            inside = not inside
    return inside

class TronEnvWithEnemy(TronBaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.enemy = SafeRandomEnemy(Position(self.width // 4, self.height // 4), field=self.field)
        self.enemy_kill_reward = config["enemy_kill_reward"]
        self.enemy_tail = []
        self.prev_angle = None
        self.angle_accum = 0.0
        self.step_closer_to_enemy_reward = config["step_closer_to_enemy_reward"]
        self.step_further_from_enemy_penalty = config["step_further_from_enemy_penalty"]


    def reset(self, seed=None, options=None):
        super().reset()
        self.enemy_tail = []
        self.enemy = SafeRandomEnemy(Position(self.width // 4, self.height // 4), field=self.field, )
        self.reward = 0
        self.prev_angle = None
        self.angle_accum = 0.0

        return self.field.state, {}


    def step(self, action):
        old_dist = abs(self.agent_pos.x - self.enemy.pos.x) + abs(self.agent_pos.y - self.enemy.pos.y)

        self.field.state, step_reward, done, truncated, info = super().step(action)
        if done or (type(action) != int and type(action) != np.int64):
            return self.field.state, step_reward, done, truncated, info

        new_dist = abs(self.agent_pos.x - self.enemy.pos.x) + abs(self.agent_pos.y - self.enemy.pos.y)

        if new_dist < old_dist and not done:
            step_reward += self.step_closer_to_enemy_reward
        elif new_dist > old_dist and not done:
            step_reward -= self.step_further_from_enemy_penalty

        enemy_move = self.enemy.move()
        old_enemy_pos = Position(self.enemy.pos.x, self.enemy.pos.y)

        match enemy_move:
            case 0:
                self.enemy.pos.y -= 1
            case 1:
                self.enemy.pos.x += 1
            case 2:
                self.enemy.pos.y += 1
            case 3:
                self.enemy.pos.x -= 1

        if len(self.enemy_tail) > 1 and (self.enemy_tail[-1].y == self.enemy.pos.y and
            self.enemy_tail[-1].x == self.enemy.pos.x):
            self.enemy.pos = Position(old_enemy_pos.x, old_enemy_pos.y)

        self.enemy_tail.append(Position(x=old_enemy_pos.x, y=old_enemy_pos.y))
        if len(self.enemy_tail) > self.tail_length:
            tail_end = self.enemy_tail.pop(0)

            self.field.update_cell(tail_end.x, tail_end.y, 0)

            self.dirty_rects.append((tail_end.y, tail_end.x))

        enemy_killed = False
        if self.field.is_out_of_bounds(self.enemy.pos.x, self.enemy.pos.y) or self.field.state[self.enemy.pos.y, self.enemy.pos.x] == 3 :
            self.enemy.pos = Position(self.width // 4, self.height // 4)
            for i, el in enumerate(self.enemy_tail):
                self.field.update_cell(el.x, el.y, 0)
                self.dirty_rects.append((el.y, el.x))
            self.enemy_tail = []
            enemy_killed = True
            #print("enemy killed by itself or wall")

        elif (self.field.state[self.enemy.pos.y, self.enemy.pos.x] == 2 or
             self.field.state[self.enemy.pos.y, self.enemy.pos.x] == 1):
            self.reward += self.enemy_kill_reward
            step_reward += self.enemy_kill_reward
            done = True

        poly = [(self.agent_pos.x, self.agent_pos.y)]

        for pos in self.tail[:-1]:
            poly.append((pos.x, pos.y))

        ex, ey = self.enemy.pos.x, self.enemy.pos.y
        if point_in_poly(ex, ey, poly):
            step_reward += 1.0
            info['surrounded_enemy'] = True

        if not done:
            if not enemy_killed:
                self.field.update_cell(old_enemy_pos.x, old_enemy_pos.y, 3)
                self.dirty_rects.append((old_enemy_pos.y, old_enemy_pos.x))

            self.field.update_cell(self.enemy.pos.x, self.enemy.pos.y, 3)
            self.dirty_rects.append((self.enemy.pos.y, self.enemy.pos.x))

        return self.field.state, step_reward, done, truncated, info


if __name__ == "__main__":
    import yaml

    with open("../configs/field_settings.yaml") as f:
        config = yaml.safe_load(f)
    pygame.init()
    env = TronEnvWithEnemy(config)
    obs, info = env.reset()
    done = False

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_ESCAPE:
                    break

            print(action)
            if type(action) == int:
                obs, reward, done, truncated, info = env.step(action)
                # print("reward:", reward)
            env.render()
            # elif type(action) == None and env.steps < 2:

        if done:
            env.reset()
    env.close()