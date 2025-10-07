import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys

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

class MonteCarloAgent:
    def __init__(self, env, episodes=5000, gamma=0.9):
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.Q = np.zeros((env.size, env.size, env.action_space.n))
        self.returns = [[ [ [] for _ in range(env.action_space.n)] for _ in range(env.size)] for _ in range(env.size)]
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self, visualize_interval=1000):
        for ep in range(self.episodes):
            state, _ = self.env.reset()
            episode = []
            done = False
            while not done:
                s = tuple(state)
                a = np.random.choice(self.env.action_space.n)
                next_state, reward, done, _, _ = self.env.step(a)
                episode.append((s, a, reward))
                state = next_state.copy()
                if done:
                    break
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r
                if (s, a) not in visited:
                    self.returns[s[0]][s[1]][a].append(G)
                    self.Q[s[0], s[1], a] = np.mean(self.returns[s[0]][s[1]][a])
                    visited.add((s, a))
            if (ep+1) % (self.episodes//5) == 0:
                print(f"Monte Carlo progress: episode {ep+1}/{self.episodes}")
        self.policy = np.argmax(self.Q, axis=2)
        print("Monte Carlo Q-values:")
        print(self.Q)
        print("Monte Carlo Policy:")
        print(self.policy)

    def get_action(self, state):
        return self.policy[state[0], state[1]]

class QLearningAgent:
    def __init__(self, env, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.size, env.size, env.action_space.n))
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self, visualize_interval=1000):
        for ep in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state[0], state[1]])
                next_state, reward, done, _, _ = self.env.step(action)
                best_next = np.max(self.Q[next_state[0], next_state[1]])
                self.Q[state[0], state[1], action] += self.alpha * (reward + self.gamma * best_next - self.Q[state[0], state[1], action])
                state = next_state.copy()
            if (ep+1) % (self.episodes//5) == 0:
                print(f"Q-Learning progress: episode {ep+1}/{self.episodes}")
        self.policy = np.argmax(self.Q, axis=2)
        print("Q-Learning Q-values:")
        print(self.Q)
        print("Q-Learning Policy:")
        print(self.policy)

    def get_action(self, state):
        return self.policy[state[0], state[1]]

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self):
        while True:
            delta = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    v = self.V[i, j]
                    values = []
                    for a in range(self.env.action_space.n):
                        pos = np.array([i, j])
                        if a == 0 and pos[0] > 0:
                            pos[0] -= 1
                        elif a == 1 and pos[0] < self.env.size - 1:
                            pos[0] += 1
                        elif a == 2 and pos[1] > 0:
                            pos[1] -= 1
                        elif a == 3 and pos[1] < self.env.size - 1:
                            pos[1] += 1
                        reward = 1 if np.array_equal(pos, self.env.goal_pos) else 0
                        values.append(reward + self.gamma * self.V[pos[0], pos[1]])
                    self.V[i, j] = max(values)
                    delta = max(delta, abs(v - self.V[i, j]))
            if delta < self.theta:
                break
        for i in range(self.env.size):
            for j in range(self.env.size):
                values = []
                for a in range(self.env.action_space.n):
                    pos = np.array([i, j])
                    if a == 0 and pos[0] > 0:
                        pos[0] -= 1
                    elif a == 1 and pos[0] < self.env.size - 1:
                        pos[0] += 1
                    elif a == 2 and pos[1] > 0:
                        pos[1] -= 1
                    elif a == 3 and pos[1] < self.env.size - 1:
                        pos[1] += 1
                    reward = 1 if np.array_equal(pos, self.env.goal_pos) else 0
                    values.append(reward + self.gamma * self.V[pos[0], pos[1]])
                self.policy[i, j] = np.argmax(values)
        print("Value Iteration Value Function:")
        print(self.V)
        print("Value Iteration Policy:")
        print(self.policy)
        visualize_agent(self.env, self, "Value Iteration Policy")

    def get_action(self, state):
        return self.policy[state[0], state[1]]

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self):
        is_policy_stable = False
        while not is_policy_stable:
            # Policy Evaluation
            while True:
                delta = 0
                for i in range(self.env.size):
                    for j in range(self.env.size):
                        a = self.policy[i, j]
                        pos = np.array([i, j])
                        if a == 0 and pos[0] > 0:
                            pos[0] -= 1
                        elif a == 1 and pos[0] < self.env.size - 1:
                            pos[0] += 1
                        elif a == 2 and pos[1] > 0:
                            pos[1] -= 1
                        elif a == 3 and pos[1] < self.env.size - 1:
                            pos[1] += 1
                        reward = 1 if np.array_equal(pos, self.env.goal_pos) else 0
                        v = self.V[i, j]
                        self.V[i, j] = reward + self.gamma * self.V[pos[0], pos[1]]
                        delta = max(delta, abs(v - self.V[i, j]))
                if delta < self.theta:
                    break
            # Policy Improvement
            policy_stable = True
            for i in range(self.env.size):
                for j in range(self.env.size):
                    old_action = self.policy[i, j]
                    values = []
                    for a in range(self.env.action_space.n):
                        pos = np.array([i, j])
                        if a == 0 and pos[0] > 0:
                            pos[0] -= 1
                        elif a == 1 and pos[0] < self.env.size - 1:
                            pos[0] += 1
                        elif a == 2 and pos[1] > 0:
                            pos[1] -= 1
                        elif a == 3 and pos[1] < self.env.size - 1:
                            pos[1] += 1
                        reward = 1 if np.array_equal(pos, self.env.goal_pos) else 0
                        values.append(reward + self.gamma * self.V[pos[0], pos[1]])
                    self.policy[i, j] = np.argmax(values)
                    if old_action != self.policy[i, j]:
                        policy_stable = False
            is_policy_stable = policy_stable
        print("Policy Iteration Value Function:")
        print(self.V)
        print("Policy Iteration Policy:")
        print(self.policy)
        visualize_agent(self.env, self, "Policy Iteration Policy")

    def get_action(self, state):
        return self.policy[state[0], state[1]]

def visualize_agent(env, agent, title, max_steps=100):
    env.reset()
    env.agent_pos = np.array([0, 0])
    done = False
    pygame.display.set_caption(title)
    steps = 0
    while not done and steps < max_steps:
        action = agent.get_action(env.agent_pos)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        steps += 1
    pygame.time.wait(1000)

if __name__ == "__main__":
    env = GridWorldEnv(size=5)
    agents = [
        (MonteCarloAgent(env), "Monte Carlo"),
        (QLearningAgent(env), "Q-Learning"),
        (ValueIterationAgent(env), "Value Iteration"),
        (PolicyIterationAgent(env), "Policy Iteration")
    ]
    for agent, name in agents:
        print(f"Training {name} agent...")
        agent.train()
        print(f"Visualizing {name} agent...")
        visualize_agent(env, agent, name, max_steps=100)
    env.close()
