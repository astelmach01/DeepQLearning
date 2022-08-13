from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from src.agents.base_agent import BaseAgent
from src.replay_buffers.basic_buffer import BasicBuffer
from src.models.basic_CNN import get_model


class DeepQLearningAgent(BaseAgent):

    def __init__(self, action_dim: int, max_memory_capacity: int,
                 in_channels: int, batch_size: int = 64, ) -> None:
        super().__init__(action_dim)

        self.discount = 0.95
        self.model = get_model(in_channels=in_channels, output_dim=action_dim)
        self.offline = get_model(in_channels=in_channels, output_dim=action_dim)

        for p in self.offline.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0005, eps=1e-4)

        self.memory = BasicBuffer(max_memory_capacity)
        self.epsilon = .1
        self.batch_size = batch_size

        self.loss = torch.nn.MSELoss()

    def store_transition(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor, done: torch.Tensor) -> None:
        """
        Stores the s, a, r, s' tuple in the replay buffer
        :param done: if the episode is terminated
        :param state: the current state of the environment, should be a Tensor
        :param action: the action taken
        :param reward: the reward given
        :param next_state: the next state of the environment based on the next action, should be a
        Tensor
        :return: None
        """
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample(self) -> Tuple[Tensor, int, float, Tensor, Tensor]:
        return self.memory.sample(self.batch_size)

    def gradient_descent(self) -> None:

        state, action, reward, next_state, done = self.sample()

        q_estimate = self.model(state)[torch.arange(0, self.batch_size), action]

        with torch.no_grad():
            best_action = torch.argmax(self.model(next_state), dim=1)

            next_q = self.offline(torch.arange(0, self.batch_size), best_action)

            q_target = (reward + (1 - done.float()) * self.epsilon * next_q).float()

        self.optimizer.zero_grad()

        loss = self.loss(q_target, q_estimate)
        loss.backward()

        self.optimizer.step()

    def act(self, state: Union[np.array, torch.Tensor]) -> int:
        """
        Picks with probability epsilon of selecting a random action a
        Or select an action from the neural network
        :param state: the current state of the environment, should be a Tensor
        :return: an integer value regarding which action to take
        """
        if torch.rand(size=(1,)) < self.epsilon:
            return torch.randint(low=0, high=self.action_dim, size=(1,)).item()

        if type(state) == np.array:
            state = torch.Tensor(state)
        return torch.argmax(self.model(state)).item()
