from collections import deque
import random
from typing import Tuple

import torch

from src.replay_buffers.a_memory import ABuffer


class BasicBuffer(ABuffer):

    def __init__(self, max_capacity: int):
        super().__init__(max_capacity)

        self.memory = deque(maxlen=max_capacity)

    def store_transition(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor, done: torch.Tensor) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, int, float, torch.Tensor, torch.Tensor]:
        state, action, reward, next_state, done = map(torch.stack,
                                                      zip(*random.sample(self.memory, batch_size)))

        return state, action, reward, next_state, done
