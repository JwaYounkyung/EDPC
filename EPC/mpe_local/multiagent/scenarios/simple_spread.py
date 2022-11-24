import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../../mpe_local'))

import itertools as it

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, sight, no_wheel, ratio):
        self.n_good = n_good
        self.n_landmarks = n_landmarks
        self.n_food = n_food
        self.n_forests = n_forests
        self.alpha = alpha
        self.sight = sight
        self.no_wheel = no_wheel
        print(sight,"sight___simple_spread_v25")
        print(alpha,"alpha######################")

    def make_world(self):
        world = World()
        # set any world properties first
        world.collaborative = True
        world.dim_c = 5
        num_good_agents = self.n_good
        world.num_good_agents = num_good_agents
        num_agents = num_good_agents
        num_landmarks = self.n_landmarks
        num_food = self.n_food
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.live = 1
            agent.time = 0
            agent.occupy = 0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.occupy = [0]
        world.landmarks += world.food
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.live = 1
            agent.time = 0
            agent.occupy = [0]
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.occupy = 0
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [self.dist(a, l) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def info(self, agent, world):
        time_grass = []
        time_live = []

        mark_grass = 0
        if agent.live:
            time_live.append(1)
            for food in world.food:
                if self.is_collision(agent, food):
                    mark_grass = 1
                    break
        else:
            time_live.append(0)
        if mark_grass:
            time_grass.append(1)
        else:
            time_grass.append(0)

        return np.concatenate([np.array(time_grass)]+[np.array(agent.occupy)])



    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        main_reward = self.reward_all_in_once(agent, world)
        return main_reward

    def reward_all_in_once(self, agent, world):
        num_agents = len(world.agents)
        reward_n = np.zeros(num_agents)
        for i, agent_new in enumerate(world.agents):
            for l in world.landmarks:
                reward_n[i] -= self.dist(agent_new, l)
            if agent_new.collide:
                for a in world.agents:
                    if self.is_collision(a, agent_new):
                        reward_n[i] -= 1
            if self.done(agent_new, world):
                reward_n[i] += 10.0
        return list(reward_n)

    def done(self, agent, world):
        cover = all(any(self.is_collision(a, l) for a in world.agents) for l in world.landmarks)
        return cover

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            distance = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
            if distance > self.sight:
                entity_pos.append([0,0,0])
            else:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                entity_pos.append([1])
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        other_live = []
        other_time = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            distance = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            # print(distance,'distance')
            # print(other.live, 'other_live')
            if distance > self.sight or (not other.live):
                other_pos.append([0,0])
                other_vel.append([0,0])
                other_live.append([0])
                other_time.append([0])
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel)
                other_live.append(np.array([other.live]))
                other_time.append(np.array([other.time]))
        result = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.live])] + entity_pos + other_pos + other_vel + other_live)
        return result
        