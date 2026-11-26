import torch
from typing import List, Tuple

class Buffer:
    def __init__(self, capacity, grid):
        self.best_trajectory = []
        self.worst_hpwl = [0, -1]
        self.roll_trajectory = []
        
        self.total_capacity = capacity * 2
        self.best_capacity = capacity
        self.roll_capacity = capacity

        self.count = 0
        self.grid = grid
        self.cache = False
    def store(self, traj: List[Tuple], hpwl=1e9, episode=None):                                
        self.cache = False
        saved = False
        if len(self.best_trajectory) < self.best_capacity:
            saved = True
            self.best_trajectory.append((hpwl, traj))
            if hpwl > self.worst_hpwl[0]:
                self.worst_hpwl = [hpwl, len(self.best_trajectory) - 1]
        else:
            if hpwl <= self.worst_hpwl[0]:
                saved = True
                self.best_trajectory[self.worst_hpwl[1]] = (hpwl, traj)
                self.worst_hpwl = [hpwl, self.worst_hpwl[1]]
                for i, (hpwl, _) in enumerate(self.best_trajectory):
                    if hpwl > self.worst_hpwl[0]:
                        self.worst_hpwl = [hpwl, i]
        
        if not saved:
            if len(self.roll_trajectory) < self.roll_capacity:
                self.roll_trajectory.append((hpwl, traj))
            else:
                if len(self.roll_trajectory) > 0:
                    self.roll_trajectory.pop(0)
                    self.roll_trajectory.append((hpwl, traj))
        
        self.count = min(self.count + 1, self.total_capacity)
        print(f"Buffer Status (Best:New): {len(self.best_trajectory)}:{len(self.roll_trajectory)}", " " * 50)
    def get_dataset(self, device):
        if self.cache:
            return \
                self.s.to(device), \
                self.t.to(device), \
                self.a.to(device), \
                self.a_logp.to(device), \
                self.r.to(device), \
                self.s_.to(device), \
                self.t_.to(device), \
                self.done.to(device)
        length = 0
        for hpwl, traj in self.best_trajectory:
            length += len(traj)
        for hpwl, traj in self.roll_trajectory:
            length += len(traj)
        self.s = torch.zeros([length, 3, self.grid, self.grid])
        self.t = torch.zeros([length], dtype=torch.long)
        self.a = torch.zeros([length, 1], dtype=torch.long)
        self.a_logp = torch.zeros([length, 1])
        self.r = torch.zeros([length, 1])
        self.s_ = torch.zeros([length, 3, self.grid, self.grid])
        self.t_ = torch.zeros([length], dtype=torch.long)
        self.done = torch.zeros([length, 1])
        index = 0
        for hpwl, traj in self.best_trajectory:
            #s_done, t, a_done, a_logp, r, s__done, t + 1, 1.0
            for traj_s, traj_t, traj_a, traj_a_logp, traj_r, traj_s_, traj_t_, traj_done in traj:
                self.s[index] = traj_s
                self.t[index] = traj_t
                self.a[index] = traj_a
                self.a_logp[index] = traj_a_logp
                self.r[index] = traj_r
                self.s_[index] = traj_s_
                self.t_[index] = traj_t_
                self.done[index] = traj_done
                index += 1
        for hpwl, traj in self.roll_trajectory:
            for traj_s, traj_t, traj_a, traj_a_logp, traj_r, traj_s_, traj_t_, traj_done in traj:
                self.s[index] = traj_s
                self.t[index] = traj_t
                self.a[index] = traj_a
                self.a_logp[index] = traj_a_logp
                self.r[index] = traj_r
                self.s_[index] = traj_s_
                self.t_[index] = traj_t_
                self.done[index] = traj_done
                index += 1
        self.cache = True
        return \
            self.s[:index].to(device), \
            self.t[:index].to(device), \
            self.a[:index].to(device), \
            self.a_logp[:index].to(device), \
            self.r[:index].to(device), \
            self.s_[:index].to(device), \
            self.t_[:index].to(device), \
            self.done[:index].to(device)
    def enough(self):
        return self.count >= (self.total_capacity / 2)
    def __len__(self):
        return self.count


