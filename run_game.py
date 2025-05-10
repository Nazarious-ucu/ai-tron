import pygame
import yaml

from envs import TronBaseEnvMultiChannel
from envs.tron_three_level_env import TronBaseEnvSimpleMultiChannel
from scripts import *
MENU_ITEMS = [
    ("Agent vs SafeRandom", run_game_with_agent),
    ("Agent vs Blocking",   run_game_with_agent),
    ("Agent vs Blocking BIG",   run_game_with_agent),
    # ("Agent vs Agent",      run_agent_vs_agent),
    # ("Agent vs Human",      run_game_with_agent),
]

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    font = pygame.font.SysFont(None, 36)
    selected = 0

    while True:
        screen.fill((30, 30, 30))
        for i, (label, _) in enumerate(MENU_ITEMS):
            color = (200,200,50) if i == selected else (200,200,200)
            txt = font.render(label, True, color)
            screen.blit(txt, (50, 50 + i*50))

        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_UP:
                    selected = (selected - 1) % len(MENU_ITEMS)
                elif ev.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(MENU_ITEMS)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    _, func = MENU_ITEMS[selected]
                    func(*mode_args[selected])


if __name__ == "__main__":
    config = yaml.safe_load(open("configs/field_settings.yaml"))
    config2 = yaml.safe_load(open("configs/conf.yaml"))
    config3 = yaml.safe_load(open("configs/new.yaml"))
    mode_args = {
      0: (config3,"scripts/tron_ppo_model10",TronBaseEnvSimpleMultiChannel, 1),
      1: (config,"scripts/tron_ppo_model6",TronBaseEnvMultiChannel, 1),
      2: (config2, "scripts/tron_ppo_model7",TronBaseEnvMultiChannel, 1),
      3: (config, "envs/selfplay_sb3/agent_round0", 1),
      # 3: ("tron_ppo_model6.zip",TronBaseEnvMultiChannel, 10,),
    }
    main_menu()
