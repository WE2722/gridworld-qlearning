import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import random

np.random.seed(123)
random.seed(123)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=5, cell_size=100):
        super().__init__()
        self.size = size
        self.cell_size = cell_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)

        self.agent_pos = None
        self.goal_pos = np.array([size-1, size-1])

        # Pygame settings
        pygame.init()
        self.window_size = size * cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("GridWorld")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        return self.agent_pos, {}

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # right
            self.agent_pos[1] += 1

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else 0
        return self.agent_pos, reward, done, False, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        # Draw grid
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw goal
        goal_rect = pygame.Rect(self.goal_pos[1]*self.cell_size, self.goal_pos[0]*self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)

        # Draw agent
        agent_rect = pygame.Rect(self.agent_pos[1]*self.cell_size, self.agent_pos[0]*self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), agent_rect)

        pygame.display.flip()
        self.clock.tick(2)  # control speed: 2 steps per second

    def close(self):
        pygame.quit()

class QLearningAgent:
    def __init__(self, env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, reward_shaping=False):
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_shaping = reward_shaping
        self.Q = np.zeros((env.size, env.size, env.action_space.n))
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self):
        for ep in range(self.episodes):
            state, _ = self.env.reset(seed=123)
            done = False
            t = 0
            while not done:
                self.epsilon = max(0.05, 0.2*100/(100+t))
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state[0], state[1]])
                next_state, reward, done, _, _ = self.env.step(action)
                if self.reward_shaping:
                    reward = reward - 0.01
                best_next = np.max(self.Q[next_state[0], next_state[1]])
                self.Q[state[0], state[1], action] += self.alpha * (reward + self.gamma * best_next - self.Q[state[0], state[1], action])
                state = next_state.copy()
                t += 1
            if (ep+1) % (self.episodes//5) == 0:
                print(f"Q-Learning progress: episode {ep+1}/{self.episodes}")
        self.policy = np.argmax(self.Q, axis=2)
        print(f"Q-Learning Q-values{' (Reward Shaping)' if self.reward_shaping else ''}:")
        print(self.Q)
        print(f"Q-Learning Policy{' (Reward Shaping)' if self.reward_shaping else ''}:")
        print(self.policy)

    def get_action(self, state):
        return self.policy[state[0], state[1]]

def visualize_agent(env, policy, title, max_steps=100):
    env.reset()
    env.agent_pos = np.array([0, 0])
    done = False
    pygame.display.set_caption(title)
    steps = 0
    clock = pygame.time.Clock()
    while not done and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = policy[env.agent_pos[0], env.agent_pos[1]]
        obs, reward, done, _, _ = env.step(action)
        env.render()
        steps += 1
        clock.tick(10)  
    pygame.time.wait(1000)

if __name__ == "__main__":
    train_envs = [GridWorldEnv(size=5), GridWorldEnv(size=5)]
    vis_env = GridWorldEnv(size=5)
    agents = [
        QLearningAgent(train_envs[0], reward_shaping=False),
        QLearningAgent(train_envs[1], reward_shaping=True)
    ]
    policies = []
    rewards = []
    names = ["Q-Learning (Standard)", "Q-Learning (Reward Shaping)"]
    for i, agent in enumerate(agents):
        print(f"Training {names[i]} agent...")
        total_reward = 0
        for ep in range(agent.episodes):
            state, _ = agent.env.reset(seed=123)
            done = False
            t = 0
            ep_reward = 0
            while not done:
                agent.epsilon = max(0.05, 0.2*100/(100+t))
                if np.random.rand() < agent.epsilon:
                    action = agent.env.action_space.sample()
                else:
                    action = np.argmax(agent.Q[state[0], state[1]])
                next_state, reward, done, _, _ = agent.env.step(action)
                if agent.reward_shaping:
                    reward = reward - 0.01
                best_next = np.max(agent.Q[next_state[0], next_state[1]])
                agent.Q[state[0], state[1], action] += agent.alpha * (reward + agent.gamma * best_next - agent.Q[state[0], state[1], action])
                state = next_state.copy()
                t += 1
                ep_reward += reward
            total_reward += ep_reward
            if (ep+1) % (agent.episodes//5) == 0:
                print(f"Q-Learning progress: episode {ep+1}/{agent.episodes}")
        agent.policy = np.argmax(agent.Q, axis=2)
        print(f"Q-Learning Q-values{' (Reward Shaping)' if agent.reward_shaping else ''}:")
        print(agent.Q)
        print(f"Q-Learning Policy{' (Reward Shaping)' if agent.reward_shaping else ''}:")
        print(agent.policy)
        policies.append(agent.policy)
        rewards.append(total_reward / agent.episodes)
        print(f"Visualizing {names[i]} agent...")
        visualize_agent(vis_env, agent.policy, names[i], max_steps=100)
    print("\nComparison of Final Policies:")
    print("Standard Q-Learning Policy:\n", policies[0])
    print("Reward Shaping Q-Learning Policy:\n", policies[1])
    print("\nAverage Episode Reward (Gain d'apprentissage):")
    print(f"Standard Q-Learning: {rewards[0]:.4f}")
    print(f"Reward Shaping Q-Learning: {rewards[1]:.4f}")
    vis_env.close()
    for env in train_envs:
        env.close()
