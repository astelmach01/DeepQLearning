from typing import Tuple

import torch


class BaseAgent:

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, state) -> int:
        raise NotImplementedError

    def store_transition(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor, done: torch.Tensor) -> None:
        raise NotImplementedError

    def sample(self) -> Tuple[torch.Tensor, int, float, torch.Tensor]:
        raise NotImplementedError

    def gradient_descent(self) -> None:
        raise NotImplementedError
