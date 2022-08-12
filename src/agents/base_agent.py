class BaseAgent:

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, state) -> int:
        raise NotImplementedError
