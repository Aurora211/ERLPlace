import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import Tensor
from tensorboardX import SummaryWriter

from models.Actor import Actor
from models.Critic import Critic
from models.Buffer import Buffer

class Agent:
    def __init__(self, benchmark_config: dict, agent_id: int = -1, ):
        super().__init__()
        self.config = benchmark_config
        self.agent_id = agent_id
        self.actor_net = Actor(benchmark_config.get('number_of_macros_to_place', 128), benchmark_config.get('grid', 256))
        self.critic_net = Critic(benchmark_config.get('number_of_macros_to_place', 128), benchmark_config.get('grid', 256))
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), benchmark_config.get('actor_learning_rate', 0.001))
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), benchmark_config.get('critic_learning_rate', 0.001))
        self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, benchmark_config.get('actor_learning_rate_anneal_rate', 0.999))
        self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, benchmark_config.get('critic_learning_rate_anneal_rate', 0.999))
        self.training_step = 0
        self.lamda = benchmark_config.get('lambda', 0.95)
        self.max_grad_norm = benchmark_config.get('maximum_gradient_normalization', 0.5)
        self.clip_epsilon = benchmark_config.get('clip_epsilon', 0.2)
        self.entropy_coef = benchmark_config.get('entropy_coefficient', 0.01)
        self.gamma = benchmark_config.get('gamma', 0.99)
        self.grid = benchmark_config.get('grid', 256)
    def set_up(self, device, tb_writer: SummaryWriter):
        self.device = device
        self.actor_net.to(device)
        self.critic_net.to(device)
        gird_X, grid_Y = torch.meshgrid(torch.linspace(0, 1, self.grid+2)[1:-1], torch.linspace(0, 1, self.grid+2)[1:-1], indexing='xy')
        self.grid_points = torch.stack([gird_X, grid_Y], dim=-1).reshape(-1, 2).to(device)
        self.tb_writer = tb_writer
        self.actor_net.grid_points = self.grid_points
    @torch.no_grad()
    def select_action(self, state: Tensor, time_step:int):
        state = state.unsqueeze(0).to(self.device)
        wire_mask = state[:,1]
        position_mask = state[:,2]
        time_step_tensor = torch.LongTensor([time_step]).to(self.device)
        self.actor_net.eval()
        distr, _, _, _ = self.actor_net(state, time_step_tensor, position_mask=position_mask, wire_mask=wire_mask)
        action = distr.sample()
        action_log_prob = distr.log_prob(action.squeeze())
        return action.item(), action_log_prob.item()
    @torch.no_grad()
    def select_action_greedy(self, state: Tensor, time_step:int):
        state = state.unsqueeze(0).to(self.device)
        wire_mask = state[:,1]
        position_mask = state[:,2]
        time_step_tensor = torch.LongTensor([time_step]).to(self.device)
        self.actor_net.eval()
        distr, _, _, _ = self.actor_net(state, time_step_tensor, position_mask=position_mask, wire_mask=wire_mask)
        action = torch.argmax(distr.probs).unsqueeze(-1)
        action_log_prob = distr.log_prob(action.squeeze())
        return action.item(), action_log_prob.item()
    @torch.no_grad()
    def act_greedy(self, state: Tensor):
        wire_mask = state[1].float().to(self.device).unsqueeze(0)
        position_mask = state[2].int().to(self.device).unsqueeze(0)
        mask = torch.where(position_mask > 0.5, torch.inf, wire_mask).reshape(self.grid * self.grid)
        wire_min = mask.min()
        action_cand = torch.where(mask == wire_min)[0]
        action_idx = torch.randint(len(action_cand), (1,)).item()
        return action_cand[action_idx].item() # type: ignore
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    def update(self, replay_buffer: Buffer):
        s, t, a, a_logp, r, s_, t_, done = replay_buffer.get_dataset(self.device)
        adv = []
        self.critic_net.eval()
        with torch.no_grad():
            v_target_list = []
            target = 0
            for re, d in zip(reversed(r.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                if d: target = 0
                target = re + self.gamma * target
                v_target_list.append(target)
            v_target_list.reverse()
            v_target = torch.tensor(v_target_list, dtype=torch.float).view(-1, 1).to(self.device)
            adv = v_target - self.critic_net(s, t)
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
            distr, _, _, _ = self.actor_net(s, t, position_mask=s[:,2], wire_mask=s[:,1])
            a_logp = distr.log_prob(a.squeeze()).unsqueeze(1)
        self.actor_net.train()
        self.critic_net.train()
        for _ in range(self.config.get('ppo_epoch', 10)):
            for idx in BatchSampler(SubsetRandomSampler(range(len(s))), self.config.get('ppo_batch_size', 128), False):
                distr, _, _, _ = self.actor_net(s[idx], t[idx], position_mask=s[idx][:,2], wire_mask=s[idx][:,1])
                distr_entropy = torch.mean(distr.entropy())
                a_logp_new = distr.log_prob(a[idx].squeeze()).unsqueeze(1)
                ratios = torch.exp(torch.clamp(a_logp_new - a_logp[idx], max=7))
                surr1 = ratios * adv[idx]
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv[idx]
                actor_loss = - torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                v_s = self.critic_net(s[idx], t[idx])
                critic_loss = F.smooth_l1_loss(v_s, v_target[idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.training_step += 1
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
                self.tb_writer.add_scalar(f'Agent_{self.agent_id}/Loss/actor_loss', actor_loss, self.training_step)
                self.tb_writer.add_scalar(f'Agent_{self.agent_id}/Loss/critic_loss', critic_loss.mean(), self.training_step)
                self.tb_writer.add_scalar(f'Agent_{self.agent_id}/Loss/entropy', distr_entropy, self.training_step)
                self.tb_writer.add_scalar(f'Agent_{self.agent_id}/Params/lr_actor', self.actor_lr_scheduler.get_last_lr()[0], self.training_step)
                self.tb_writer.add_scalar(f'Agent_{self.agent_id}/Params/lr_critic', self.critic_lr_scheduler.get_last_lr()[0], self.training_step)


