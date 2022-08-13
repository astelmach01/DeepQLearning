from typing import Tuple

import torch


class ABuffer:

    def __init__(self, max_capacity: int):
        self.max_capacity = max_capacity

    def store_transition(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor, done: torch.Tensor) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, int, float, torch.Tensor]:
        raise NotImplementedError
