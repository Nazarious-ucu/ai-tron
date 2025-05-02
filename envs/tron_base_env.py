import numpy as np
from envs.base_env import BaseTronEnv

from envs.Enteties.position import Position
import pygame



class TronBaseEnv(BaseTronEnv):

    def __init__(self, config):
        super().__init__(config)
        self.reward = 0
        self.agent_pos = None
        self.tail = []
        self.steps = 0
        self.screen = None
        self.clock = None

        self.dirty_rects = []
        self.lose_by_enemy = config["lose_by_enemy"]
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset()

        self.agent_pos = Position(self.height // 2, self.width // 2)
        self.field.update_cell(self.agent_pos.x, self.agent_pos.y, 1)

        self.tail = []
        self.steps = 0
        self.screen = None

        self.reward = 0
        return self.field.state, {}

    def step(self, action):
        old_y, old_x = self.agent_pos.y, self.agent_pos.x
        done_game = False

        self.tail.append(Position(x=old_x, y=old_y))
        self.reward += self.reward_for_step
        # print("action: ",action, type(action))
        # self.field.update_cell(old_x, old_y, 0)

        if len(self.tail) > self.tail_length:
            tail_end = self.tail.pop(0)

            self.field.update_cell(tail_end.x, tail_end.y, 0)

            self.dirty_rects.append((tail_end.y, tail_end.x))

        match action:
            case 0:
                self.agent_pos.y -= 1
            case 1:
                self.agent_pos.x += 1
            case 2:
                self.agent_pos.y += 1
            case 3:
                self.agent_pos.x -= 1

            case _:
                # print("Undefined action")
                ...


        if (self.field.is_out_of_bounds(self.agent_pos.x, self.agent_pos.y) or
            self.field.state[self.agent_pos.y, self.agent_pos.x] == 2):
            done_game = True
            self.reward -= self.penalty_for_death
            # print("you killed by wall")


        elif self.field.state[self.agent_pos.y, self.agent_pos.x] == 3:
            done_game = True
            self.reward -= self.lose_by_enemy
            # print("you killed by enemy")

        else :
            self.field.update_cell(old_x, old_y, 2)

            self.field.update_cell(self.agent_pos.x, self.agent_pos.y, 1)


        self.steps += 1

        # if self.steps % 50 == 0:
        #     self.reward += 10

        # if self.steps % 200 == 0:
        #     self.reward += 100


        self.dirty_rects.append((old_y, old_x))
        self.dirty_rects.append((self.agent_pos.y, self.agent_pos.x))

        truncated = False
        return self.field.state, self.reward, done_game, truncated, {}

    def render(self, done = False):
        if self.screen is None:
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption('Tron Environment')
            self.clock = pygame.time.Clock()

            for y in range(self.height):
                for x in range(self.width):
                    if self.field.state[y, x] == -1:
                        color = (200, 50, 50)
                        rect = pygame.Rect(
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.display.update(rect)

        if not done:
            rect_list = []
            for (y, x) in self.dirty_rects:
                cell_value = self.field.state[y, x]
                if cell_value == 1:
                    color = (0, 255, 0)
                elif cell_value == 2:
                    color = (0, 100, 255)
                elif cell_value == 3:
                    color = (0, 0, 255)
                else:
                    color = (50, 50, 50)

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                rect_list.append(rect)

            pygame.display.update(rect_list)

        self.dirty_rects.clear()

        # self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


if __name__ == "__main__":
    import yaml

    with open("../configs/field_settings.yaml") as f:
        config = yaml.safe_load(f)
    pygame.init()
    env = TronBaseEnv(config)
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
                    action = 2  # down
                elif event.key == pygame.K_LEFT:
                    action = 3  # left
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_ESCAPE:
                    break


            obs, reward, done, truncated, info = env.step(action)
            # print("reward:", reward)
            env.render()
        if done:
            env.reset()
    env.close()
