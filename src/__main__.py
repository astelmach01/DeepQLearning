import gym

from src.agents.deepq_learning_agent import DeepQLearningAgent


def loop(env: gym.ObservationWrapper, agent, num_episodes=40000, checkpoint_period=20,
         render=True):
    episode = 0
    for _ in range(num_episodes):
        try:
            state = env.reset()
            done = False
            reward_per_episode = 0
            while not done:  # what happens during every episode

                action = agent.act(state)

                if render or episode > 400:
                    env.render()

                next_state, reward, done, info = env.step(action)

                agent.store_transition(state, next_state, action, reward, done)
                agent.gradient_descent()

                state = next_state
                reward_per_episode += reward

        except KeyboardInterrupt:
            render = not render
            continue


def sweat(render=True):
    env = gym.make("CartPole-v1")

    agent = DeepQLearningAgent(action_dim=env.action_space.n, in_channels=4,
                               max_memory_capacity=50000)
    num_episodes = 40000

    loop(env, agent, num_episodes, render=render)


sweat()
