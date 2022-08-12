import random
from collections import deque

from typing import Tuple
# noinspection PyPackageRequirements

import torch

from base_agent import BaseAgent


class DeepQLearningAgent(BaseAgent):

    def __init__(self, action_dim: int, batch_size: int = 64) -> None:
        super().__init__(action_dim)

        self.model = self.get_model()
        self.offline = self.get_model()

        self.memory = deque()
        self.epsilon = .1
        self.batch_size = batch_size

    def act(self, state: torch.Tensor) -> int:
        """
        Picks with probability epsilon of selecting a random action a
        Or select an action from the neural network
        :param state: the current state of the environment, should be a Tensor
        :return: an integer value regarding which action to take
        """
        if torch.rand(size=(1,)) < self.epsilon:
            return torch.randint(low=0, high=self.action_dim, size=(1,)).item()

        return torch.argmax(self.model(state)).item()

    def store_transition(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor) -> None:
        """
        Stores the s, a, r, s' tuple in the replay buffer
        :param state: the current state of the environment, should be a Tensor
        :param action: the action taken
        :param reward: the reward given
        :param next_state: the next state of the environment based on the next action, should be a
        Tensor
        :return: None
        """
        self.memory.append((state, action, reward, next_state))

    def sample(self) -> Tuple[torch.Tensor, int, float, torch.Tensor]:
        state, action, reward, next_state = map(torch.stack,
                                                zip(*random.sample(self.memory, self.batch_size)))

        return state, action, reward, next_state

    def gradient_descent(self) -> None:
        pass
