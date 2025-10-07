import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys


class GridWorldEnv:
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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # main agent
        self.agent_pos = None

        # goals
        self.n_goals = max(0, int(n_goals))
        self.goals_dynamic = bool(goals_dynamic)
        self.goals = []
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
        self._provided_obstacles = bool(obstacles_pos)
        self._init_obstacles = []
        if obstacles_pos:
            for p in obstacles_pos:
                arr = np.array(p, dtype=int)
                self.obstacles.append(arr)
                self._init_obstacles.append(arr.copy())

        # other agents
        self.n_other_agents = max(0, int(n_other_agents))
        self.other_agents_dynamic = bool(other_agents_dynamic)
        self.other_agents = []
        self._provided_other_agents = bool(other_agents_pos)
        self._init_other_agents = []
        if other_agents_pos:
            for p in other_agents_pos:
                arr = np.array(p, dtype=int)
                self.other_agents.append(arr)
                self._init_other_agents.append(arr.copy())

        # Pygame settings
        self.render_enabled = bool(render)
        self.screen = None
        self.clock = None
        if self.render_enabled:
            try:
                import pygame
                pygame.init()
                self.window_size = (cols * cell_size, rows * cell_size)
                self.screen = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("GridWorld")
                self.clock = pygame.time.Clock()
            except Exception as e:
                print(f"Warning: rendering disabled; could not initialize pygame display: {e}")
                self.render_enabled = False
                self.screen = None
                self.clock = None

        # maximum steps per episode
        if max_steps is None:
            self.max_steps = max(100, rows * cols * 4)
        else:
            self.max_steps = int(max_steps)

    def _in_bounds(self, pos):
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def _random_empty_cell(self, forbidden):
        forbidden_set = {(int(p[0]), int(p[1])) for p in forbidden}
        choices = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in forbidden_set]
        if not choices:
            return np.array([0, 0], dtype=int)
        r, c = random.choice(choices)
        return np.array([r, c], dtype=int)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Place main agent at top-left
        self.agent_pos = np.array([0, 0], dtype=int)
        forbidden = [self.agent_pos]

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

        # Initialize obstacles
        self.obstacles = []
        if hasattr(self, 'n_obstacles') and self.n_obstacles > 0:
            if getattr(self, '_provided_obstacles', False) and len(self._init_obstacles) > 0:
                for o in self._init_obstacles:
                    if any(np.array_equal(o, f) for f in forbidden):
                        raise ValueError(f"Obstacle position {o} conflicts with existing entity!")
                    self.obstacles.append(o.copy())
                    forbidden.append(o.copy())
            else:
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
        action = random.choice([0, 1, 2, 3, None])
        if action is None:
            return pos
        return self._move_pos(pos, action)

    def step(self, action):
        """Execute one step in the environment."""
        # Calculate new position
        new_pos = self._move_pos(self.agent_pos, action)
        
        # Check bounds and obstacles
        hit_wall = not self._in_bounds(new_pos)
        hit_obstacle = any(np.array_equal(new_pos, o) for o in self.obstacles)
        
        # Apply movement or stay in place
        if hit_wall or hit_obstacle:
            new_pos = self.agent_pos.copy()
            bump_penalty = -0.1
        else:
            bump_penalty = 0
        
        self.agent_pos = new_pos

        # Move dynamic entities
        if self.other_agents_dynamic:
            for i, p in enumerate(self.other_agents):
                self.other_agents[i] = self._random_move(p)

        if self.goals_dynamic:
            for i, p in enumerate(self.goals):
                self.goals[i] = self._random_move(p)

        if self.obstacles_dynamic:
            for i, p in enumerate(self.obstacles):
                self.obstacles[i] = self._random_move(p)

        # Check terminal conditions
        for g in self.goals:
            if np.array_equal(self.agent_pos, g):
                return self.agent_pos.copy(), 1, True, False, {}

        for a in self.other_agents:
            if np.array_equal(self.agent_pos, a):
                return self.agent_pos.copy(), -1, True, False, {}

        return self.agent_pos.copy(), bump_penalty, False, False, {}

    def render(self):
        if not self.render_enabled:
            return
        import pygame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        # Draw grid lines
        for i in range(self.rows):
            for j in range(self.cols):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw obstacles (black)
        for o in self.obstacles:
            rect = pygame.Rect(o[1] * self.cell_size, o[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

        # Draw goals (red)
        for g in self.goals:
            rect = pygame.Rect(g[1] * self.cell_size, g[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw other agents (orange)
        for a in self.other_agents:
            rect = pygame.Rect(a[1] * self.cell_size, a[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 200, 0), rect)

        # Draw main agent (bright green)
        agent_rect = pygame.Rect(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), agent_rect)

        pygame.display.flip()
        if self.clock:
            self.clock.tick(5)

    def close(self):
        if self.render_enabled:
            import pygame
            pygame.quit()


# DQN Network
class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# Experience Replay Buffer
class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, env, episodes=500, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, memory_size=10000, target_update=10):
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        state_dim = 2  # (row, col)
        action_dim = 4  # up, down, left, right
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
        
        # Tracking
        self.rewards_history = []
        self.lengths_history = []
        self.losses_history = []
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, verbose=True):
        """Train the DQN agent"""
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            done = False
            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                
                # Store transition
                self.memory.push(state, action, reward, next_state, float(done))
                
                # Train
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Check max steps
                if episode_length >= getattr(self.env, 'max_steps', 1000):
                    done = True
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Track metrics
            self.rewards_history.append(episode_reward)
            self.lengths_history.append(episode_length)
            if episode_losses:
                self.losses_history.append(np.mean(episode_losses))
            
            if verbose and (episode + 1) % max(1, self.episodes // 10) == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                avg_length = np.mean(self.lengths_history[-10:])
                print(f"Episode {episode+1}/{self.episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def get_action(self, state):
        """Get action for a given state (greedy)"""
        return self.select_action(state, training=False)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")


def visualize_agent(env, agent, max_steps=100):
    """Visualize the trained agent navigating the environment"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grid visualization
    def draw_grid(ax, state, goals, obstacles, other_agents):
        ax.clear()
        ax.set_xlim(-0.5, env.cols - 0.5)
        ax.set_ylim(-0.5, env.rows - 0.5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.invert_yaxis()
        
        # Draw obstacles
        for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                       color='black', alpha=0.8))
        
        # Draw goals
        for goal in goals:
            ax.add_patch(plt.Circle((goal[1], goal[0]), 0.3, 
                                    color='red', alpha=0.7))
        
        # Draw other agents
        for other in other_agents:
            ax.add_patch(plt.Circle((other[1], other[0]), 0.25, 
                                    color='orange', alpha=0.7))
        
        # Draw main agent
        ax.add_patch(plt.Circle((state[1], state[0]), 0.35, 
                                color='green', alpha=0.9))
        
        ax.set_title('Agent Navigation', fontsize=14, fontweight='bold')
    
    # Initialize
    state, _ = env.reset()
    trajectory = [state.copy()]
    rewards_over_time = []
    cumulative_reward = 0
    
    # Run episode
    done = False
    step = 0
    while not done and step < max_steps:
        action = agent.get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        state = next_state
        trajectory.append(state.copy())
        cumulative_reward += reward
        rewards_over_time.append(cumulative_reward)
        step += 1
    
    print(f"\nEpisode finished in {len(trajectory)} steps with reward {cumulative_reward:.2f}")
    
    # Animation
    def animate(frame):
        if frame < len(trajectory):
            state = trajectory[frame]
            draw_grid(ax1, state, env.goals, env.obstacles, env.other_agents)
            
            # Plot cumulative reward
            ax2.clear()
            ax2.plot(rewards_over_time[:frame+1], 'b-', linewidth=2)
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Cumulative Reward', fontsize=12)
            ax2.set_title(f'Cumulative Reward (Step {frame}/{len(trajectory)-1})', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                        interval=300, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return trajectory, rewards_over_time


def plot_training_metrics(agent):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    axes[0, 0].plot(agent.rewards_history, alpha=0.6, label='Episode Reward')
    window = min(50, max(1, len(agent.rewards_history) // 10))
    if window > 1:
        moving_avg = np.convolve(agent.rewards_history, 
                                np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(agent.rewards_history)), 
                       moving_avg, 'r-', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(agent.lengths_history, alpha=0.6, label='Episode Length')
    if window > 1:
        moving_avg = np.convolve(agent.lengths_history, 
                                np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(agent.lengths_history)), 
                       moving_avg, 'r-', linewidth=2, label=f'MA({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss
    if agent.losses_history:
        axes[1, 0].plot(agent.losses_history, alpha=0.6, label='Loss')
        if window > 1:
            moving_avg = np.convolve(agent.losses_history, 
                                    np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(agent.losses_history)), 
                           moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Success rate
    success_window = 50
    successes = [1 if r > 0 else 0 for r in agent.rewards_history]
    if len(successes) >= success_window:
        success_rate = [np.mean(successes[max(0, i-success_window):i+1]) 
                       for i in range(len(successes))]
        axes[1, 1].plot(success_rate, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title(f'Success Rate (Window={success_window})')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create GridWorld environment
    env = GridWorldEnv(
        rows=5,
        cols=5,
        n_goals=1,
        goals_pos=[[4, 4]],  # Goal at bottom-right
        n_obstacles=2,
        obstacles_pos=[[2, 2], [2, 3]],  # Two obstacles in the middle
        render=False,  # Disable pygame during training
        max_steps=100
    )
    
    print("=" * 60)
    print("Training DQN Agent on GridWorld")
    print("=" * 60)
    
    agent = DQNAgent(
        env=env,
        episodes=300,
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=10
    )
    
    # Train the agent
    agent.train(verbose=True)
    
    # Plot training metrics
    print("\n" + "=" * 60)
    print("Plotting training metrics...")
    print("=" * 60)
    plot_training_metrics(agent)
    
    # Visualize the trained agent
    print("\n" + "=" * 60)
    print("Visualizing trained agent...")
    print("=" * 60)
    visualize_agent(env, agent, max_steps=100)
    
    # Save the model
    agent.save_model('dqn_gridworld.pth')
    
    env.close()