import os
import random
import logging
import torch
import pickle
import math
import numpy as np
from typing import List, Tuple, Dict
from tensorboardX import SummaryWriter

from models import Agent
from models.Buffer import Buffer
from models.Layer import ACTOR_LAYERS, CRITIC_LAYERS
from envs import Environment_ISPD2005
from trainer.solution_pool import SolutionPool

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, benchmark_config: Dict, job_folder: str):
        self.benchmark_config       = benchmark_config
        self.num_macros_to_place    = benchmark_config.get("number_of_macros_to_place", 128)
        self.buffer_capacity        = benchmark_config.get("buffer_capacity", 10)
        self.update_frontiers_begin = benchmark_config.get("update_frontiers_begin", 20)
        self.update_frontiers_freq  = benchmark_config.get("update_frontiers_freq", 2)
        self.solution_pool          = SolutionPool(benchmark_config.get("solution_pool_size", 10), benchmark_config.get("solution_pool_alpha", 0.3))
        self.model_dir              = os.path.join(job_folder, "models")
        self.solution_dir           = os.path.join(job_folder, "solutions")
        self.placement_dir          = os.path.join(job_folder, "placements")
        os.makedirs(self.model_dir,     exist_ok=True)
        os.makedirs(self.solution_dir,  exist_ok=True)
        os.makedirs(self.placement_dir, exist_ok=True)
        # Essential Global Variables
        self.global_episode = 0
    def setup_tb_writer(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer
    def train(self, population: List[Tuple[int, Agent, Dict]], env: Environment_ISPD2005, device: torch.device):
        logger.info(f"===== ERL Training =========================================")

        replay_buffer = Buffer(self.buffer_capacity, env.grid)
        
        loop = 0
        population_state_dicts: List[Dict] = []
        for (agent_id, agent, _) in population:
            state_dicts = agent.actor_net.state_dict()
            population_state_dicts.append({"id": agent_id, "dict": state_dicts})
        
        while self.global_episode < self.benchmark_config.get("max_episodes", 1000):
            loop += 1
            logger.info("----- Epoch {:<5d} ------------------------------------------".format(loop))
            # Evaluation ======================================================================================================================================
            for (agent_id, agent, statistics) in population:
                # Evaluation ==================================================================================================================================
                hpwl, sol_id, solution, placement, invalid_count = self.run_episode(env, agent, replay_buffer)
                hpwl_item = hpwl.item()
                logger.info(f"Episode {self.global_episode}, Agent {agent_id}, HPWL = {hpwl}, Invalid Actions = {invalid_count}")
                # Global Search Tree Update ===================================================================================================================
                self.solution_pool.update_solution(sol_id, solution, hpwl, placement)
                # Rollback Process ============================================================================================================================
                if loop >= self.benchmark_config.get("layer_crossover_begin", 5) and loop % self.benchmark_config.get("layer_crossover_interval", 5) == 1:
                    if statistics['crossover']:
                        last_hpwl = statistics['last']
                        if hpwl_item > last_hpwl + self.benchmark_config.get("layer_rollback_threshold", 0.5):
                            if agent_id != population_state_dicts[agent_id]["id"]:
                                logger.critical(f"Agent ID mismatch: {agent_id} != {population_state_dicts[agent_id]['id']}")
                                raise ValueError(f"Agent ID mismatch: {agent_id} != {population_state_dicts[agent_id]['id']}")
                            agent.actor_net.load_state_dict(population_state_dicts[agent_id]["dict"].copy())
                            statistics['rollback'] = True
                            logger.info(f"Agent {agent_id} rolled back.")
                        else:
                            statistics['rollback'] = False
                    else:
                        statistics['rollback'] = False
                else:
                    statistics['rollback'] = False
                # Agent Statistics Update =====================================================================================================================
                statistics['elite'] = False
                statistics['last'] = hpwl_item
                # Tensorboard Logging =========================================================================================================================
                self.tb_writer.add_scalar('epochs/hpwl', hpwl_item, self.global_episode)
                self.tb_writer.add_scalar("epochs/best_hpwl", self.solution_pool.best_hpwl, self.global_episode)
                self.tb_writer.add_scalar("epochs/best_hpwl_max", self.solution_pool.best_hpwl_max, self.global_episode)
                self.tb_writer.add_scalar("epochs/depth_min", self.solution_pool.depth_min_max[0], self.global_episode)
                self.tb_writer.add_scalar("epochs/depth_max", self.solution_pool.depth_min_max[1], self.global_episode)
            # Frontiers Update ================================================================================================================================
            if loop >= self.update_frontiers_begin and loop % self.update_frontiers_freq == 0:
                self.solution_pool.update_frontiers()
            # Best Solution Saving ============================================================================================================================
            placement, _, hpwl = self.solution_pool.best_solution
            logger.info(f"Current best HPWL = {hpwl}.")
            os.makedirs(self.solution_dir, exist_ok=True)
            with open(os.path.join(self.solution_dir, f"best_placement_{hpwl}.pkl"), 'wb') as f:
                pickle.dump(placement, f)
            pl_file_name = os.path.join(self.placement_dir, f"best_placement_{hpwl}.pl")
            if not os.path.exists(pl_file_name):
                verified_hpwl = env.save(placement, pl_file_name)
                logger.info(f"PL file saved. Verified HPWL = {verified_hpwl / 1e5}.")
            # Agent Ranking ===================================================================================================================================
            agent_ranks = sorted(range(len(population)), key=lambda x: population[x][2]['last'])
            elite_agent_indexs = agent_ranks[:self.benchmark_config.get("elite_count", 1)]
            for i in range(len(population)):
                if i in elite_agent_indexs:
                    population[i][2]['elite'] = True
                else:
                    population[i][2]['elite'] = False
            # Agent Crossover & Mutation ======================================================================================================================
            if loop >= self.benchmark_config.get('layer_crossover_begin', 5) and loop % self.benchmark_config.get('layer_crossover_interval', 5) == 0:
                # Agent Rollback Preparation ==================================================================================================================
                for (agent_id, agent, statistics) in population:
                    # Agent Crossover Rollback Preparation
                    agent_state_dict = agent.actor_net.state_dict().copy()
                    if population_state_dicts[agent_id]["id"] != agent_id:
                        logger.critical(f"Agent ID mismatch: {population_state_dicts[agent_id]['id']} != {agent_id}")
                        raise ValueError(f"Agent ID mismatch: {population_state_dicts[agent_id]['id']} != {agent_id}")
                    population_state_dicts[agent_id]["dict"] = agent_state_dict
                # Agent Crossover =============================================================================================================================
                hpwls = [statistics['last'] for (_, _, statistics) in population]
                normalized_hpwl = [(hpwl - min(hpwls)) / (max(hpwls) - min(hpwls)) * self.benchmark_config.get('ea_crossover_normalize_coefficient', 2) * -1 for hpwl in hpwls]
                exp_hpwl = [np.exp(hpwl) for hpwl in normalized_hpwl]
                for (agent_id, agent, statistics) in population:
                    if statistics['elite']:
                        continue
                    # Softmin Selection
                    exp_hpwl_for_agent = exp_hpwl.copy()
                    exp_hpwl_for_agent[agent_id] = 0
                    exp_hpwl_for_agent_sum = np.sum(exp_hpwl_for_agent)
                    probabilities = [exp_hpwl / exp_hpwl_for_agent_sum for exp_hpwl in exp_hpwl_for_agent]
                    selected_id = np.random.choice(range(len(population)), p=probabilities)
                    target_agent_id, target_agent, target_agent_statistics = population[selected_id]
                    if target_agent_id != selected_id:
                        logger.critical(f"Agent ID mismatch: {target_agent_id} != {selected_id}")
                        raise ValueError(f"Agent ID mismatch: {target_agent_id} != {selected_id}")
                    # Agent Crossover Update
                    actor_layers = [layer['prefix'] for layer in ACTOR_LAYERS.copy()]
                    for layer in actor_layers:
                        for name, param in agent.actor_net.named_parameters():
                            if layer in name:
                                target_param = target_agent.actor_net.state_dict()[name]
                                mask = torch.rand_like(param.data) < self.benchmark_config.get("layer_crossover_prob", 0.5)
                                new_data = torch.where(mask, target_param.data, param.data)
                                param.data.copy_(new_data)
                                if mask.any():
                                    statistics['crossover'] = True
                    critic_layers = [layer['prefix'] for layer in CRITIC_LAYERS.copy()]
                    for layer in critic_layers:
                        for name, param in agent.critic_net.named_parameters():
                            if layer in name:
                                target_param = target_agent.critic_net.state_dict()[name]
                                mask = torch.rand_like(param.data) < self.benchmark_config.get("layer_crossover_prob", 0.5)
                                new_data = torch.where(mask, target_param.data, param.data)
                                param.data.copy_(new_data)
                                if mask.any():
                                    statistics['crossover'] = True
                    if statistics['crossover']:
                        logger.info(f"Agent {agent_id} crossed with Agent {target_agent_id}.")
            else:
                for (agent_id, agent, statistics) in population:
                    statistics["crossover"] = False
                    if replay_buffer.enough():
                        agent.update(replay_buffer)
                        logger.info(f"Agent {agent_id} reinforment updated.")
    def run_episode(self, env: Environment_ISPD2005, agent: Agent, replay_buffer: Buffer):
        # initialize the environment
        buffer = []
        t, hpwl, score = 0, 0, 0
        solution, placement = [], []
        invalid_action = 0
        s = env.reset()
        done = False
        # randomly select a frontier to start an episode from
        _, sol = self.solution_pool.sample()
        while True:
            if t < len(sol):
                # select the action from the solution
                a, a_logp = sol[t], 0.0
                print(f"Agent: False, Node ID: {t:>5d}, Action: {a:>6d}, Node Size: {env.macros[t]['x'] * env.macros[t]['y']}", end="\r")
            else:
                a, a_logp = agent.select_action(s, t)
                print(f"Agent: True , Node ID: {t:>5d}, Action: {a:>6d}, Node Size: {env.macros[t]['x'] * env.macros[t]['y']}", end="\r")
            s_, r, done, info = env.step(a)
            placement.append(a)
            hpwl += info["delta_hpwl"]
            if info['invalid_action']:
                invalid_action += 1
            if not done:
                if t >= len(sol):
                    #replay_buffer.store(s, t, a, a_logp, r, s_, t + 1, 0.0)
                    buffer.append((s, t, a, a_logp, r, s_, t + 1, 0.0))
                solution.append(a)
                score += r
                t += 1
            if done and env.num_macros > self.num_macros_to_place:
                # if the episode is done, place the remaining macros greedily
                s_done, s__done, a_done = s, s_, a
                s = s_
                for i in range(env.num_macros - self.num_macros_to_place):
                    a = agent.act_greedy(s)
                    s_, r_, done, info = env.step(a)
                    placement.append(a)
                    hpwl += info["delta_hpwl"]
                    if info['invalid_action']:
                        invalid_action += 1
                    r += r_
                    if i == env.num_macros - self.num_macros_to_place - 1:
                        #replay_buffer.store(s_done, t, a_done, a_logp, r, s__done, t + 1, 1.0)
                        buffer.append((s_done, t, a_done, a_logp, r, s__done, t + 1, 1.0))
                        solution.append(a_done)
                    s = s_
                score += r
                break
            s = s_
        if replay_buffer is not None:
            replay_buffer.store(buffer, hpwl * env.ratio / 1e5)
        self.global_episode += 1
        return hpwl * env.ratio / 1e5, 0, solution, placement, invalid_action


