from envs.Enemies.chasing_enemy import ChasingEnemy
from envs.Enteties.position import Position


class BlockerEnemy:
    def __init__(self, start, field):
        self.pos, self.field = start, field

    def move(self, agent_pos, agent_action):
        dx, dy = [(0,-1),(1,0),(0,1),(-1,0)][agent_action]
        targ = Position(agent_pos.x+1*dx, agent_pos.y+1*dy)

        return ChasingEnemy(self.pos,self.field).move(targ)
