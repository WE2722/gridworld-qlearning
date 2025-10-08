try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    print("Required package 'gymnasium' is not installed. Install dependencies with: pip install -r requirements.txt")
    raise

import threading
import streamlit as st
import time
import numpy as np
import pygame
import sys
import random
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw

np.random.seed(123)
random.seed(123)

class GridWorldEnv(gym.Env):
    """
    Flexible GridWorld environment.

    Parameters (new):
    - rows, cols: grid dimensions
    - n_goals, goals_pos, goals_dynamic
    - n_obstacles, obstacles_pos, obstacles_dynamic
    - n_other_agents, other_agents_pos, other_agents_dynamic
    - cell_size: pixels for rendering

    Colors: agents => green, goals => red, obstacles => black
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rows=5,
        cols=5,
        cell_size=100,
        n_goals=1,
        goals_pos=None,
        goals_dynamic=False,
        n_obstacles=0,
        obstacles_pos=None,
        obstacles_dynamic=False,
        n_other_agents=0,
        other_agents_pos=None,
        other_agents_dynamic=False,
        render=True,
        max_steps=None,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=max(rows, cols)-1, shape=(2,), dtype=np.int32)

        # main agent
        self.agent_pos = None

        # goals
        self.n_goals = max(0, int(n_goals))
        self.goals_dynamic = bool(goals_dynamic)
        self.goals = []
        # store provided initial goal positions so reset can keep them fixed
        self._provided_goals = bool(goals_pos)
        self._init_goals = []
        if goals_pos:
            for p in goals_pos:
                arr = np.array(p, dtype=int)
                self.goals.append(arr)
                self._init_goals.append(arr.copy())

        # obstacles
        self.n_obstacles = max(0, int(n_obstacles))
        self.obstacles_dynamic = bool(obstacles_dynamic)
        self.obstacles = []
        # store provided obstacle positions so reset can keep them fixed
        self._provided_obstacles = bool(obstacles_pos)
        self._init_obstacles = []
        if obstacles_pos:
            for p in obstacles_pos:
                arr = np.array(p, dtype=int)
                self.obstacles.append(arr)
                self._init_obstacles.append(arr.copy())

        # other agents (competitors that can stop main agent)
        self.n_other_agents = max(0, int(n_other_agents))
        self.other_agents_dynamic = bool(other_agents_dynamic)
        self.other_agents = []
        # store provided other-agent positions so reset can keep them fixed
        self._provided_other_agents = bool(other_agents_pos)
        self._init_other_agents = []
        if other_agents_pos:
            for p in other_agents_pos:
                arr = np.array(p, dtype=int)
                self.other_agents.append(arr)
                self._init_other_agents.append(arr.copy())

        # Pygame settings (only when rendering enabled)
        self.render_enabled = bool(render)
        self.screen = None
        self.clock = None
        if self.render_enabled:
            try:
                pygame.init()
                self.window_size = (cols * cell_size, rows * cell_size)
                self.screen = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("GridWorld")
                self.clock = pygame.time.Clock()
            except Exception as e:
                # If display initialization fails (no DISPLAY or other OS issue), disable rendering
                print(f"Warning: rendering disabled; could not initialize pygame display: {e}")
                self.render_enabled = False
                self.screen = None
                self.clock = None

        # maximum steps per episode (depends on grid if not provided)
        if max_steps is None:
            # default scale: proportional to number of cells
            self.max_steps = max(100, rows * cols * 4)
        else:
            self.max_steps = int(max_steps)

    def _in_bounds(self, pos):
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def _random_empty_cell(self, forbidden):
        # forbidden is a list of positions (arrays or tuples)
        forbidden_set = { (int(p[0]), int(p[1])) for p in forbidden }
        choices = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r,c) not in forbidden_set]
        if not choices:
            return np.array([0,0], dtype=int)
        r,c = random.choice(choices)
        return np.array([r, c], dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Place main agent at top-left OR find random empty cell if top-left is occupied
        forbidden = []
        
        # First, initialize obstacles if provided (they take precedence for position blocking)
        self.obstacles = []
        if hasattr(self, 'n_obstacles') and self.n_obstacles > 0:
            if getattr(self, '_provided_obstacles', False) and len(self._init_obstacles) > 0:
                for o in self._init_obstacles:
                    self.obstacles.append(o.copy())
                    forbidden.append(o.copy())
            # Random obstacles will be placed after agent
        
        # Place agent - avoid obstacles
        agent_start = np.array([0, 0], dtype=int)
        if any(np.array_equal(agent_start, f) for f in forbidden):
            # Top-left is blocked, find random empty cell
            agent_start = self._random_empty_cell(forbidden)
        self.agent_pos = agent_start
        forbidden.append(self.agent_pos)

        # Initialize goals
        self.goals = []
        if hasattr(self, 'n_goals') and self.n_goals > 0:
            if getattr(self, '_provided_goals', False) and len(self._init_goals) > 0:
                for g in self._init_goals:
                    if any(np.array_equal(g, f) for f in forbidden):
                        raise ValueError(f"Goal position {g} conflicts with existing entity!")
                    self.goals.append(g.copy())
                    forbidden.append(g.copy())
            else:
                for _ in range(self.n_goals):
                    cell = self._random_empty_cell(forbidden)
                    self.goals.append(cell)
                    forbidden.append(cell)

        # Initialize remaining random obstacles (if any)
        if hasattr(self, 'n_obstacles') and self.n_obstacles > 0:
            if not (getattr(self, '_provided_obstacles', False) and len(self._init_obstacles) > 0):
                for _ in range(self.n_obstacles):
                    cell = self._random_empty_cell(forbidden)
                    self.obstacles.append(cell)
                    forbidden.append(cell)

        # Initialize other agents
        self.other_agents = []
        if hasattr(self, 'n_other_agents') and self.n_other_agents > 0:
            if getattr(self, '_provided_other_agents', False) and len(self._init_other_agents) > 0:
                for a in self._init_other_agents:
                    if any(np.array_equal(a, f) for f in forbidden):
                        raise ValueError(f"Agent position {a} conflicts with existing entity!")
                    self.other_agents.append(a.copy())
                    forbidden.append(a.copy())
            else:
                for _ in range(self.n_other_agents):
                    cell = self._random_empty_cell(forbidden)
                    self.other_agents.append(cell)
                    forbidden.append(cell)

        return self.agent_pos.copy(), {}

    def _move_pos(self, pos, action):
        new = pos.copy()
        if action == 0 and new[0] > 0:  # up
            new[0] -= 1
        elif action == 1 and new[0] < self.rows - 1:  # down
            new[0] += 1
        elif action == 2 and new[1] > 0:  # left
            new[1] -= 1
        elif action == 3 and new[1] < self.cols - 1:  # right
            new[1] += 1
        return new

    def _random_move(self, pos):
        action = random.choice([0,1,2,3, None])
        if action is None:
            return pos
        return self._move_pos(pos, action)
    
    def _random_move_safe(self, pos, forbidden_positions):
        """Move randomly but avoid colliding with forbidden positions.
        
        Args:
            pos: current position
            forbidden_positions: list of positions to avoid (e.g., agent position, other entities)
        
        Returns:
            new position (stays in place if all moves would cause collision)
        """
        # Try all possible moves including staying in place
        possible_moves = [0, 1, 2, 3, None]
        random.shuffle(possible_moves)
        
        for action in possible_moves:
            if action is None:
                new_pos = pos
            else:
                new_pos = self._move_pos(pos, action)
            
            # Check if this position is safe (not in forbidden list and in bounds)
            if self._in_bounds(new_pos):
                collision = False
                for forbidden in forbidden_positions:
                    if np.array_equal(new_pos, forbidden):
                        collision = True
                        break
                
                if not collision:
                    return new_pos
        
        # If no safe move found, stay in place
        return pos

    def step(self, action):
        """
        Execute one step in the environment.
        
        IMPORTANT BEHAVIOR:
        1. Agent bumps into obstacles: stays in place, gets small penalty (-0.1), continues
        2. Agent tries to move outside grid: stays in place, gets small penalty (-0.1), continues
        3. Colliding with other agents: large penalty (-1), episode terminates
        4. Reaching goal: reward (+1), episode terminates
        5. Dynamic entities avoid moving onto the agent's position or each other
        """
        # Calculate new position based on action
        new_pos = self._move_pos(self.agent_pos, action)
        
        # Check if new position is out of bounds (hitting wall)
        hit_wall = not self._in_bounds(new_pos)
        
        # Check if new position hits an obstacle
        hit_obstacle = False
        for o in self.obstacles:
            if np.array_equal(new_pos, o):
                hit_obstacle = True
                break
        
        # If hitting wall or obstacle, stay in place and apply small penalty
        if hit_wall or hit_obstacle:
            new_pos = self.agent_pos.copy()
            bump_penalty = -0.1
        else:
            bump_penalty = 0
        
        # Move agent to new position
        self.agent_pos = new_pos

        # Move dynamic entities, ensuring they don't move onto agent's position or collide
        # Collect all current entity positions to avoid collisions
        forbidden_for_obstacles = [self.agent_pos]
        forbidden_for_goals = [self.agent_pos]
        forbidden_for_agents = [self.agent_pos]
        
        # Move obstacles dynamically (if enabled) - they avoid agent
        if self.obstacles_dynamic:
            new_obstacles = []
            for i, o in enumerate(self.obstacles):
                # Each obstacle avoids agent and other obstacles that already moved
                forbidden = forbidden_for_obstacles + new_obstacles
                new_pos = self._random_move_safe(o, forbidden)
                new_obstacles.append(new_pos)
            self.obstacles = new_obstacles

        # Update forbidden positions for goals
        forbidden_for_goals.extend(self.obstacles)
        
        # Move goals dynamically (if enabled) - they avoid agent and obstacles
        if self.goals_dynamic:
            new_goals = []
            for i, g in enumerate(self.goals):
                forbidden = forbidden_for_goals + new_goals
                new_pos = self._random_move_safe(g, forbidden)
                new_goals.append(new_pos)
            self.goals = new_goals

        # Update forbidden positions for other agents
        forbidden_for_agents.extend(self.obstacles)
        forbidden_for_agents.extend(self.goals)
        
        # Move other agents dynamically (if enabled) - they avoid everything
        if self.other_agents_dynamic:
            new_agents = []
            for i, a in enumerate(self.other_agents):
                forbidden = forbidden_for_agents + new_agents
                new_pos = self._random_move_safe(a, forbidden)
                new_agents.append(new_pos)
            self.other_agents = new_agents

        # Check terminal conditions AFTER all movement
        # Reached any goal - positive reward and termination
        for g in self.goals:
            if np.array_equal(self.agent_pos, g):
                return self.agent_pos.copy(), 1, True, False, {}

        # Collided with other agent - large penalty and termination
        for a in self.other_agents:
            if np.array_equal(self.agent_pos, a):
                return self.agent_pos.copy(), -1, True, False, {}

        # Normal step - apply bump penalty if any, no termination
        return self.agent_pos.copy(), bump_penalty, False, False, {}
    def render(self):
        if not self.render_enabled:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        # Draw grid lines
        for i in range(self.rows):
            for j in range(self.cols):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw obstacles (black)
        for o in self.obstacles:
            rect = pygame.Rect(o[1]*self.cell_size, o[0]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

        # Draw goals (red)
        for g in self.goals:
            rect = pygame.Rect(g[1]*self.cell_size, g[0]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw other agents (green)
        for a in self.other_agents:
            rect = pygame.Rect(a[1]*self.cell_size, a[0]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 200, 0), rect)

        # Draw main agent (green)
        agent_rect = pygame.Rect(self.agent_pos[1]*self.cell_size, self.agent_pos[0]*self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), agent_rect)

        pygame.display.flip()
        if self.clock:
            self.clock.tick(5)  # control speed

    def close(self):
        if self.render_enabled:
            pygame.quit()

class QLearningAgent:
    def __init__(self, env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, reward_shaping=False):
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_shaping = reward_shaping
        # Q-table shaped by environment rows x cols x actions
        self.Q = np.zeros((env.rows, env.cols, env.action_space.n))
        self.policy = np.zeros((env.rows, env.cols), dtype=int)

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
                if t >= getattr(self.env, 'max_steps', 1000):
                    done = True
            if (ep+1) % (self.episodes//5) == 0:
                print(f"Q-Learning progress: episode {ep+1}/{self.episodes}")
        self.policy = np.argmax(self.Q, axis=2)
        print(f"Q-Learning Q-values{' (Reward Shaping)' if self.reward_shaping else ''}:")
        print(self.Q)
        print(f"Q-Learning Policy{' (Reward Shaping)' if self.reward_shaping else ''}:")
        print(self.policy)

    def train_track(self, episodes=None, tol=1e-6, seed=123):
        """Train while tracking per-episode reward and Q-table change magnitude.

        Returns: rewards(list), q_deltas(list)
        """
        if episodes is None:
            episodes = self.episodes
        rewards = []
        deltas = []
        lengths = []
        for ep in range(episodes):
            state, _ = self.env.reset(seed=seed)
            done = False
            t = 0
            ep_reward = 0
            Q_prev = self.Q.copy()
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
                if t >= getattr(self.env, 'max_steps', 1000):
                    done = True
                ep_reward += reward
            rewards.append(ep_reward)
            lengths.append(t)
            delta = np.max(np.abs(self.Q - Q_prev))
            deltas.append(delta)
            # early stop if converged
            if delta < tol:
                # truncate lists to episode count
                return rewards, deltas, lengths
            if (ep+1) % (episodes//5) == 0:
                print(f"Q-Learning progress: episode {ep+1}/{episodes}")
        return rewards, deltas, lengths

    def get_action(self, state):
        return self.policy[state[0], state[1]]

    def save_model(self, path):
        """Save the Q-table and policy to disk as .npy files (path prefix)."""
        try:
            np.save(path + '_q.npy', self.Q)
            np.save(path + '_policy.npy', self.policy)
            print(f"Saved model files: {path}_q.npy, {path}_policy.npy")
        except Exception as e:
            print('Failed to save model:', e)

    @staticmethod
    def load_model(path):
        """Load and return a QLearningAgent-like dict with Q and policy from path prefix."""
        q = np.load(path + '_q.npy')
        policy = np.load(path + '_policy.npy')
        return {'Q': q, 'policy': policy}

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
        # guard against policy shape mismatch
        r,c = int(env.agent_pos[0]), int(env.agent_pos[1])
        if r < policy.shape[0] and c < policy.shape[1]:
            action = int(policy[r, c])
        else:
            action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        steps += 1
        clock.tick(10)  
    pygame.time.wait(1000)


def visualize_convergence(rewards, deltas, title="Convergence"):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.plot(rewards, color='tab:blue', label='Episode Reward')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Q delta')
    ax2.plot(deltas, color='tab:orange', label='Max Q delta')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    plt.title(title)
    plt.show()


def visualize_gamma_sensitivity(env_factory, gammas, episodes=200):
    """Run experiments for different gamma values. env_factory is a callable that returns a fresh env instance."""
    conv_iters = []
    avg_rewards = []
    for g in gammas:
        env = env_factory()
        agent = QLearningAgent(env, episodes=episodes, gamma=g)
        rewards, deltas, lengths = agent.train_track(episodes=episodes)
        # measure convergence episode
        conv_ep = len(deltas)
        conv_iters.append(conv_ep)
        # mean number of iterations (steps) per episode as proxy for iterations-to-converge
        mean_iters = np.mean(lengths) if lengths else 0
        avg_rewards.append(np.mean(rewards))
        print(f"gamma={g}: conv_ep={conv_ep}, mean_iters={mean_iters:.2f}, avg_reward={np.mean(rewards):.4f}")
        env.close()

    # (Intentionally do not save per-episode files here; function produces the two summary plots above.)

    fig, ax1 = plt.subplots()
    ax1.plot(gammas, conv_iters, '-o', label='Convergence Episode')
    ax1.set_xlabel('Gamma')
    ax1.set_ylabel('Convergence Episode')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(gammas, avg_rewards, '-s', color='tab:orange', label='Average Reward')
    ax2.set_ylabel('Average Episode Reward')
    ax2.legend(loc='upper right')
    plt.title('Gamma Sensitivity: Convergence Episode & Avg Reward')
    try:
        plt.savefig('gamma_sensitivity_conv_vs_reward.png')
        print('Saved gamma_sensitivity_conv_vs_reward.png')
    except Exception:
        pass
    plt.show()

    # Also plot mean iterations-per-episode vs gamma
    plt.figure()
    plt.plot(gammas, conv_iters, '-o', label='Convergence Episode')
    plt.xlabel('Gamma')
    plt.ylabel('Convergence Episode')
    plt.twinx()
    plt.plot(gammas, [np.mean([0]) for _ in gammas], alpha=0)  # placeholder to keep structure
    plt.title('Convergence episode vs Gamma')
    try:
        plt.savefig('gamma_sensitivity_conv.png')
        print('Saved gamma_sensitivity_conv.png')
    except Exception:
        pass
    plt.legend()
    plt.show()


def convergence_vs_grid(grid_sizes, base_goal_rel=(1.0,1.0), obstacle_rel_positions=None, episodes=300, n_runs=3, conv_tol=1e-4):
    """Measure convergence episode (number of episodes until Q-table max-delta < conv_tol)
    as a function of grid dimension. Runs `n_runs` independent repeats per grid size and plots
    mean +/- std of convergence episodes.

    - grid_sizes: iterable of ints (n for nxn grids)
    - base_goal_rel: (row_frac, col_frac) for placing the goal relative to grid (0..1)
    - obstacle_rel_positions: list of (row_frac, col_frac) for obstacles
    - episodes: max episodes per run
    - n_runs: repeats per grid size
    - conv_tol: convergence threshold on max Q change
    """
    conv_means = []
    conv_stds = []

    for item in grid_sizes:
        conv_list = []
        # allow either an int n (interpreted as n x n) or a (rows, cols) tuple
        if isinstance(item, int):
            rows = cols = item
        else:
            try:
                rows, cols = int(item[0]), int(item[1])
            except Exception:
                raise ValueError('grid_sizes must contain ints or (rows,cols) pairs')
        # compute integer positions from relative fractions
        goal_pos = [[min(rows-1, max(0, int(round(base_goal_rel[0]*(rows-1))))), min(cols-1, max(0, int(round(base_goal_rel[1]*(cols-1)))))] ]
        obstacles_pos = []
        if obstacle_rel_positions:
            for (rf, cf) in obstacle_rel_positions:
                obstacles_pos.append([min(rows-1, max(0, int(round(rf*(rows-1))))), min(cols-1, max(0, int(round(cf*(cols-1)))) )])

        last_rewards = last_deltas = last_lengths = None
        for run in range(n_runs):
            seed = 1000 + run
            env = GridWorldEnv(rows=rows, cols=cols, n_goals=1, goals_pos=goal_pos, goals_dynamic=False,
                               n_obstacles=len(obstacles_pos), obstacles_pos=obstacles_pos, obstacles_dynamic=False,
                               n_other_agents=0, render=False, seed=seed)
            agent = QLearningAgent(env, episodes=episodes)
            _, deltas, _ = agent.train_track(episodes=episodes, tol=conv_tol, seed=seed)
            # conv ep is length of deltas (if early-stop happened) otherwise episodes
            conv_ep = len(deltas)
            conv_list.append(conv_ep)
            env.close()

        conv_means.append(np.mean(conv_list))
        conv_stds.append(np.std(conv_list))
    print(f"grid={rows}x{cols}: conv_ep_mean={conv_means[-1]:.2f} std={conv_stds[-1]:.2f} over {n_runs} runs")

    # plot mean +/- std for convergence episode vs grid size
    plt.figure()
    # prepare x positions and labels so we can support rectangular grids
    x = list(range(len(grid_sizes)))
    labels = []
    for item in grid_sizes:
        if isinstance(item, int):
            labels.append(str(item))
        else:
            labels.append(f"{int(item[0])}x{int(item[1])}")

    plt.errorbar(x, conv_means, yerr=conv_stds, fmt='-o', capsize=5)
    plt.xticks(x, labels)
    plt.xlabel('Grid dimension (rows x cols)')
    plt.ylabel('Convergence Episode (mean ± std)')
    plt.title('Convergence episode vs Grid Dimension')
    plt.grid(True)
    try:
        plt.savefig('convergence_vs_grid.png')
        print('Saved convergence_vs_grid.png')
    except Exception:
        pass
    plt.show()
    # (Function only produces mean +/- std convergence vs grid size plot)


def visualize_learning_process(env, agent, episodes=100, conv_tol=1e-4, delay=0.05, live_plot=False):
    """Run training episodes while rendering the environment and updating a live learning plot.

    - env must be created with render=True so pygame rendering occurs.
    - agent will be updated in-place (Q and policy).
    - delay is seconds between environment steps to make visualization human-readable.

    Returns (rewards, deltas)
    """
    if not getattr(env, 'render_enabled', False):
        print("Warning: env.render_enabled is False — pass render=True when creating the env to visualize.")

    rewards = []
    deltas = []
    # Only prepare matplotlib interactive plotting if requested explicitly
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        try:
            plt.show(block=False)
            fig.canvas.draw()
        except Exception:
            pass
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        line_r, = ax.plot([], [], color='tab:blue', label='Episode Reward')
        ax2 = ax.twinx()
        ax2.set_ylabel('Max Q delta')
        line_d, = ax2.plot([], [], color='tab:orange', label='Max Q delta')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    for ep in range(episodes):
        state, _ = env.reset(seed=None)
        done = False
        t = 0
        ep_reward = 0
        Q_prev = agent.Q.copy()
        while not done and t < getattr(env, 'max_steps', 1000):
            agent.epsilon = max(0.05, agent.epsilon)
            if np.random.rand() < agent.epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(agent.Q[state[0], state[1]]))
            next_state, reward, done, _, _ = env.step(action)
            if agent.reward_shaping:
                reward -= 0.01
            best_next = np.max(agent.Q[next_state[0], next_state[1]])
            agent.Q[state[0], state[1], action] += agent.alpha * (reward + agent.gamma * best_next - agent.Q[state[0], state[1], action])
            state = next_state.copy()
            ep_reward += reward
            t += 1
            # render environment
            env.render()
            # small delay to make it watchable
            try:
                pygame.time.delay(int(delay*1000))
            except Exception:
                pass

        rewards.append(ep_reward)
        delta = np.max(np.abs(agent.Q - Q_prev))
        deltas.append(delta)
        agent.policy = np.argmax(agent.Q, axis=2)

        if live_plot:
            line_r.set_data(range(len(rewards)), rewards)
            line_d.set_data(range(len(deltas)), deltas)
            ax.relim(); ax.autoscale_view()
            ax2.relim(); ax2.autoscale_view()
            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                plt.pause(0.01)

        if delta < conv_tol:
            print(f"Converged at episode {ep+1} (delta {delta:.3e})")
            break

    # If plotting was created, save a snapshot; but for live pygame mode we avoid matplotlib entirely.
    if live_plot:
        try:
            fig.savefig('convergence_plot.png')
            print('Saved convergence_plot.png')
        except Exception:
            pass

        # Also save clearer individual plots for rewards and q-deltas with markers so
        # they are visible even if there are very few episodes.
        try:
            # Rewards
            plt.figure()
            if len(rewards) == 0:
                plt.plot([0], [0], 'o', color='tab:blue')
                plt.annotate('no data', xy=(0,0), xytext=(0.5, 0.5))
                plt.xlim(-1, 2)
            else:
                plt.plot(range(1, len(rewards)+1), rewards, '-o', color='tab:blue')
                if len(rewards) == 1:
                    plt.xlim(0, 2)
            plt.xlabel('Episode')
            plt.ylabel('Episode Reward')
            plt.title('Episode Rewards')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('rewards_plot.png')
            print('Saved rewards plot to rewards_plot.png')

            # Q-deltas
            plt.figure()
            if len(deltas) == 0:
                plt.plot([0], [1e-8], 'o', color='tab:orange')
                plt.xlim(-1, 2)
                plt.yscale('log')
            else:
                # avoid log-scale failure when all deltas are zero by flooring to a tiny positive value
                safe_deltas = [max(d, 1e-12) for d in deltas]
                plt.plot(range(1, len(safe_deltas)+1), safe_deltas, '-o', color='tab:orange')
                if len(safe_deltas) == 1:
                    plt.xlim(0, 2)
            plt.xlabel('Episode')
            plt.ylabel('Max Q delta')
            try:
                plt.yscale('log')
            except Exception:
                pass
            plt.title('Max Q-table Delta per Episode')
            plt.grid(True, which='both')
            plt.tight_layout()
            plt.savefig('qdelta_plot.png')
            print('Saved Q-delta plot to qdelta_plot.png')
        except Exception as e:
            print('Failed to save individual plots:', e)
        plt.ioff()
        try:
            plt.show()
        except Exception:
            pass
    return rewards, deltas


# removed save_standard_plots; this module focuses on the two requested experiments and live visualization


def export_training_gif(env, agent, episodes=50, out_path='training_vis.gif', fps=5, cell_size=None, seed=123):
    """Export a GIF showing the agent learning over `episodes` episodes.

    This runs a brief training loop similar to `visualize_learning_process` but
    renders frames using Pillow so it works headless and does not require pygame.

    - env: GridWorldEnv instance (render flag can be False)
    - agent: QLearningAgent instance (will be updated in-place)
    - episodes: total training episodes to run while capturing frames
    - out_path: output GIF filename (written in workspace folder)
    - fps: frames per second for GIF
    - cell_size: pixel size per cell for exported images (defaults to env.cell_size or 80)
    - seed: RNG seed for reproducibility
    """
    try:
        import imageio
    except Exception:
        raise RuntimeError("imageio is required to export GIFs. Install with: pip install imageio pillow")

    if cell_size is None:
        cell_size = getattr(env, 'cell_size', 80)

    random.seed(seed)
    np.random.seed(seed)

    frames = []
    width = env.cols * cell_size
    height = env.rows * cell_size

    for ep in range(episodes):
        state, _ = env.reset(seed=seed+ep)
        done = False
        t = 0
        Q_prev = agent.Q.copy()
        while not done and t < getattr(env, 'max_steps', 1000):
            # epsilon-greedy action
            agent.epsilon = max(0.05, agent.epsilon)
            if np.random.rand() < agent.epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(agent.Q[state[0], state[1]]))

            next_state, reward, done, _, _ = env.step(action)
            best_next = np.max(agent.Q[next_state[0], next_state[1]])
            agent.Q[state[0], state[1], action] += agent.alpha * (reward + agent.gamma * best_next - agent.Q[state[0], state[1], action])
            state = next_state.copy()
            t += 1

            # render a PIL image frame from current env state
            img = Image.new('RGB', (width, height), color=(255,255,255))
            draw = ImageDraw.Draw(img)

            # draw grid lines
            for r in range(env.rows):
                for c in range(env.cols):
                    x0 = c * cell_size
                    y0 = r * cell_size
                    x1 = x0 + cell_size
                    y1 = y0 + cell_size
                    draw.rectangle([x0, y0, x1, y1], outline=(200,200,200), width=1)

            # obstacles -> black
            for o in env.obstacles:
                r,c = int(o[0]), int(o[1])
                x0 = c * cell_size; y0 = r * cell_size
                draw.rectangle([x0, y0, x0+cell_size, y0+cell_size], fill=(0,0,0))

            # goals -> red
            for g in env.goals:
                r,c = int(g[0]), int(g[1])
                x0 = c * cell_size; y0 = r * cell_size
                draw.rectangle([x0, y0, x0+cell_size, y0+cell_size], fill=(255,0,0))

            # other agents -> darker green
            for a in env.other_agents:
                r,c = int(a[0]), int(a[1])
                x0 = c * cell_size; y0 = r * cell_size
                draw.rectangle([x0+cell_size*0.15, y0+cell_size*0.15, x0+cell_size*0.85, y0+cell_size*0.85], fill=(0,150,0))

            # main agent -> bright green
            ag = env.agent_pos
            r,c = int(ag[0]), int(ag[1])
            x0 = c * cell_size; y0 = r * cell_size
            draw.ellipse([x0+cell_size*0.15, y0+cell_size*0.15, x0+cell_size*0.85, y0+cell_size*0.85], fill=(0,255,0))

            frames.append(np.array(img))

        # small indication of episode boundary: append one extra frame
        frames.append(np.array(img))

    # write GIF
    try:
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Wrote GIF to {out_path} ({len(frames)} frames, fps={fps})")
    except Exception as e:
        raise RuntimeError(f"Failed to write GIF: {e}")



def run_live_training(rows=5, cols=5, goal_pos=None, obstacles_pos=None, episodes=200, cell_size=80, delay=0.05):
    """Helper: create a rendered env and agent and run the live visualization.

    Example:
      run_live_training(rows=5, cols=5, goal_pos=[[4,4]], obstacles_pos=[[1,2],[2,3]], episodes=200)
    """
    if goal_pos is None:
        goal_pos = [[rows-1, cols-1]]
    if obstacles_pos is None:
        obstacles_pos = []
    env = GridWorldEnv(rows=rows, cols=cols, n_goals=1, goals_pos=goal_pos, goals_dynamic=False,
                       n_obstacles=len(obstacles_pos), obstacles_pos=obstacles_pos, obstacles_dynamic=False,
                       n_other_agents=0, render=True, cell_size=cell_size, seed=123)
    agent = QLearningAgent(env, episodes=episodes)
    if not env.render_enabled:
        env.close()
        raise RuntimeError("Rendering is not available in this environment.")
    try:
        # For live pygame visualization, avoid matplotlib interference by disabling live_plot
        visualize_learning_process(env, agent, episodes=episodes, delay=delay, live_plot=False)
    finally:
        env.close()


def export_training_gif_custom_colors(env, agent, episodes=50, out_path='training_vis.gif', fps=5, 
                                      cell_size=None, seed=123, colors=None):
    """Export a GIF with custom colors showing the agent learning over episodes.
    
    Args:
        env: GridWorldEnv instance
        agent: QLearningAgent instance (will be updated in-place)
        episodes: total training episodes
        out_path: output GIF filename
        fps: frames per second
        cell_size: pixel size per cell
        seed: RNG seed
        colors: dict with keys 'agent', 'goal', 'obstacle', 'other_agent' (hex colors)
    """
    try:
        import imageio
        from PIL import Image, ImageDraw
    except Exception:
        raise RuntimeError("imageio and PIL are required. Install with: pip install imageio pillow")

    if cell_size is None:
        cell_size = getattr(env, 'cell_size', 80)
    
    # Default colors or use provided
    if colors is None:
        colors = {
            'agent': '#00FF00',
            'goal': '#FF0000',
            'obstacle': '#000000',
            'other_agent': '#009600'
        }
    
    # Convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    agent_rgb = hex_to_rgb(colors['agent'])
    goal_rgb = hex_to_rgb(colors['goal'])
    obstacle_rgb = hex_to_rgb(colors['obstacle'])
    other_agent_rgb = hex_to_rgb(colors['other_agent'])

    random.seed(seed)
    np.random.seed(seed)

    frames = []
    width = env.cols * cell_size
    height = env.rows * cell_size

    for ep in range(episodes):
        state, _ = env.reset(seed=seed+ep)
        done = False
        t = 0
        Q_prev = agent.Q.copy()
        
        while not done and t < getattr(env, 'max_steps', 1000):
            # epsilon-greedy action
            agent.epsilon = max(0.05, agent.epsilon)
            if np.random.rand() < agent.epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(agent.Q[state[0], state[1]]))

            next_state, reward, done, _, _ = env.step(action)
            best_next = np.max(agent.Q[next_state[0], next_state[1]])
            agent.Q[state[0], state[1], action] += agent.alpha * (
                reward + agent.gamma * best_next - agent.Q[state[0], state[1], action]
            )
            state = next_state.copy()
            t += 1

            # render a PIL image frame from current env state
            img = Image.new('RGB', (width, height), color=(255,255,255))
            draw = ImageDraw.Draw(img)

            # draw grid lines
            for r in range(env.rows):
                for c in range(env.cols):
                    x0 = c * cell_size
                    y0 = r * cell_size
                    x1 = x0 + cell_size
                    y1 = y0 + cell_size
                    draw.rectangle([x0, y0, x1, y1], outline=(200,200,200), width=1)

            # obstacles
            for o in env.obstacles:
                r, c = int(o[0]), int(o[1])
                x0 = c * cell_size
                y0 = r * cell_size
                draw.rectangle([x0, y0, x0+cell_size, y0+cell_size], fill=obstacle_rgb)

            # goals
            for g in env.goals:
                r, c = int(g[0]), int(g[1])
                x0 = c * cell_size
                y0 = r * cell_size
                draw.rectangle([x0, y0, x0+cell_size, y0+cell_size], fill=goal_rgb)

            # other agents
            for a in env.other_agents:
                r, c = int(a[0]), int(a[1])
                x0 = c * cell_size
                y0 = r * cell_size
                draw.rectangle([x0+cell_size*0.15, y0+cell_size*0.15, 
                              x0+cell_size*0.85, y0+cell_size*0.85], fill=other_agent_rgb)

            # main agent (circle)
            ag = env.agent_pos
            r, c = int(ag[0]), int(ag[1])
            x0 = c * cell_size
            y0 = r * cell_size
            draw.ellipse([x0+cell_size*0.15, y0+cell_size*0.15, 
                         x0+cell_size*0.85, y0+cell_size*0.85], fill=agent_rgb)

            frames.append(np.array(img))

        # Episode boundary frame
        frames.append(np.array(img))

    # write GIF
    try:
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Wrote GIF to {out_path} ({len(frames)} frames, fps={fps})")
    except Exception as e:
        raise RuntimeError(f"Failed to write GIF: {e}")

if __name__ == "__main__":
    # Run only the two visualizations requested (headless):
    # 1) convergence episode vs grid dimension
    # 2) sensitivity of convergence episode to gamma

    # Parameters for grid-dimension experiment
    grid_sizes = [3, 5, 7, 9]
    obstacle_rel_positions = [(1/4, 2/5), (2/5, 3/5)]
    convergence_vs_grid(grid_sizes, base_goal_rel=(1.0, 1.0), obstacle_rel_positions=obstacle_rel_positions, episodes=100, n_runs=3, conv_tol=1e-4)

    # Gamma sensitivity experiment (headless)
    rows = cols = 5
    goal_pos = [[rows-1, cols-1]]
    obstacles_pos = [[1,2],[2,3]]
    def factory():
        return GridWorldEnv(rows=rows, cols=cols, n_goals=1, goals_pos=goal_pos, goals_dynamic=False,
                           n_obstacles=len(obstacles_pos), obstacles_pos=obstacles_pos, obstacles_dynamic=False,
                           n_other_agents=0, render=False, seed=None)
    gammas = [0.0, 0.3, 0.6, 0.9, 0.99]
    visualize_gamma_sensitivity(factory, gammas, episodes=150)

    # To visualize the learning process and environment live, create a rendered env and call:
    # render_env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], n_obstacles=2, obstacles_pos=[[1,2],[2,3]], render=True, seed=123)
    # agent = QLearningAgent(render_env, episodes=200)
    # visualize_learning_process(render_env, agent, episodes=200, delay=0.05)


