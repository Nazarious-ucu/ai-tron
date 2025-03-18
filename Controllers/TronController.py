import asyncio
import time
import pygame

from envs import TronBaseEnv


class TronController:
    def __init__(self, env: TronBaseEnv, step_interval=0.5):
        self.env = env
        self.step_interval = step_interval
        self.direction = 0
        self.done = False

        self.obs, self.info = self.env.reset()

    def set_direction(self, direction: int):
        self.direction = direction

    async def  run_game_loop(self):
        self.done = False
        self.obs, self.info = self.env.reset()

        while not self.done:
            self.obs, reward, self.done, truncated, _ = self.env.step(self.direction)
            self.env.render()

            if self.done or truncated:
                print(f"Game over! Reward = {reward}")
                break
            await asyncio.sleep(self.step_interval)

    async def event_loop(self):
        while not self.done:
            await asyncio.sleep(0.01)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                    elif event.key == pygame.K_UP:
                        self.set_direction(0)
                    elif event.key == pygame.K_RIGHT:
                        self.set_direction(1)
                    elif event.key == pygame.K_DOWN:
                        self.set_direction(2)
                    elif event.key == pygame.K_LEFT:
                        self.set_direction(3)



if __name__ == "__main__":
    async def main():
        env = TronBaseEnv(width=10, height=10, cell_size=10)
        controller = TronController(env, step_interval=0.1)

        controller.set_direction(0)
        await asyncio.gather(
            controller.run_game_loop(),
            controller.event_loop()
        )
        env.close()


    asyncio.run(main())