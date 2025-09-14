import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple
from dataclasses import dataclass
from q_network import QNet


def move_optimizer_state(optimizer: optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


@dataclass
class Transition:
    s: np.ndarray  # state
    a: int         # action
    r: float       # reward
    sp: np.ndarray # next state
    done: bool     # episode terminated?


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buf: List[Transition] = []
        self.idx = 0

    def push(self, t: Transition):
        if len(self.buf) < self.capacity:
            self.buf.append(t)
        else:
            self.buf[self.idx] = t
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


class LinearEpsilon:
    def __init__(self, start=1.0, end=0.05, decay_episodes=200):
        self.start = start
        self.end = end
        self.decay = decay_episodes

    def value(self, ep: int) -> float:
        frac = min(1.0, ep / self.decay)
        return self.start + (self.end - self.start) * frac


class DQNAgent:
    def __init__(self, state_dim: int, hidden: int = 64, lr: float = 1e-3, buffer_capacity: int = 50000, target_update_every: int = 200, gamma: float = 0.99):
        self.state_dim = state_dim
        self.gamma = gamma
        self.policy_net = QNet(state_dim, hidden).to("cpu")  # keep VRAM usage low
        self.target_net = QNet(state_dim, hidden).to("cpu")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.target_update_every = target_update_every
        self.update_step = 0
        self._last_loss = None

    def push_transition(self, t: Transition):
        self.replay.push(t)

    def _active_on(self, device: torch.device):
        class _Ctx:
            def __init__(self, outer, device):
                self.outer = outer
                self.device = device
            def __enter__(self):
                self.outer.policy_net.to(self.device)
                self.outer.target_net.to(self.device)
                move_optimizer_state(self.outer.optimizer, self.device)
            def __exit__(self, exc_type, exc, tb):
                self.outer.policy_net.to("cpu")  # return to CPU to keep memory low
                self.outer.target_net.to("cpu")
                move_optimizer_state(self.outer.optimizer, torch.device("cpu"))
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return _Ctx(self, device)

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, epsilon: float, device: torch.device) -> int:
        # epsilon-greedy on the *online* network
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, 2))
        s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with self._active_on(device):
            q = self.policy_net(s)  # [1, 2]
            a = int(torch.argmax(q, dim=1).item())
        return a

    def q_values(self, state_vec: np.ndarray, device: torch.device) -> Tuple[float, float]:
        # Return Q(s,0), Q(s,1) with no grad (for logging)
        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            with self._active_on(device):
                q = self.policy_net(s).squeeze(0)  # [2]
                
                return float(q[0].item()), float(q[1].item())

    def learn(self, batch_size: int, device: torch.device):
        if len(self.replay) < batch_size:
            self._last_loss = None
            
            return None

        batch = self.replay.sample(batch_size)
        states = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b.a for b in batch], dtype=torch.long)
        rewards = torch.tensor([b.r for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([b.sp for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b.done for b in batch], dtype=torch.bool)

        with self._active_on(device):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # Q(s,a) for taken actions (online net)
            q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN target computation
            with torch.no_grad():
                # 1) action selection by ONLINE net (argmax over next-state Qs)
                next_online_q = self.policy_net(next_states)
                next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)  # [B,1]

                # 2) action evaluation by TARGET net (gather the chosen actions)
                next_target_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # [B]

                targets = rewards + (~dones).float() * self.gamma * next_target_q

            loss = self.criterion(q_sa, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.update_step += 1
            if self.update_step % self.target_update_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self._last_loss = float(loss.item())
            
            return self._last_loss
