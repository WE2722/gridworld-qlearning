# 🤖 GridWorld Q-Learning Reinforcement Learning Project

A comprehensive GridWorld environment with Q-Learning agent training, featuring an interactive Streamlit interface for easy configuration and visualization.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## ✨ Features

- **Interactive Streamlit Web Interface**: Configure all parameters through a user-friendly GUI
- **Customizable GridWorld Environment**: Configure grid size, goals, obstacles, and other agents
- **Q-Learning Implementation**: Classic reinforcement learning with epsilon-greedy exploration
- **Training Visualization**: Animated GIF export of the learning process with customizable colors
- **Model Persistence**: Save and load trained Q-tables and policies
- **Comprehensive Analysis**:
  - Grid size impact on convergence
  - Gamma (discount factor) sensitivity analysis
  - Models saved for different gamma values
- **Training Reports**: Automatic generation of detailed training reports with interpretations
- **Organized Output Structure**: All outputs saved in organized folders

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How the App Works](#how-the-app-works)
- [Reward Structure](#reward-structure)
- [Streamlit Interface Guide](#streamlit-interface-guide)
- [Output Files](#output-files)
- [Understanding the Visualizations](#understanding-the-visualizations)
- [Training Reports](#training-reports)
- [Advanced Usage](#advanced-usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/gridworld-qlearning.git
cd gridworld-qlearning
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
gymnasium>=0.29.0
numpy>=1.24.0
pygame>=2.5.0
matplotlib>=3.7.0
imageio>=2.31.0
Pillow>=10.0.0
streamlit>=1.28.0
```

### Quick Setup Script

**Linux/macOS:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

**Windows:**
```bash
quickstart.bat
```

## 📁 Project Structure

```
gridworld-qlearning/
│
├── app.py                              # Streamlit web interface
├── Livrable_3_2_WIAME_EL_HAFID.py     # Core environment & Q-Learning
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── USER_GUIDE.md                      # Detailed user guide
<<<<<<< HEAD
├── GITHUB_SETUP.md                    # GitHub setup instructions
=======
>>>>>>> c508dcdad35659ecf690dcd614a41bb5b702b04f
├── quickstart.sh                      # Setup script (Linux/macOS)
├── quickstart.bat                     # Setup script (Windows)
│
└── output/                            # Generated outputs
    ├── models/                        # Saved Q-tables and policies
    ├── plots/                         # Training visualizations
    ├── gifs/                          # Training animations
    └── reports/                       # Training reports
```

## 🎯 Quick Start

### Using Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

This will open a web browser with an interactive interface where you can:
- Configure the environment visually
- Set training parameters with sliders
- Choose which outputs to generate
- View results in real-time with detailed interpretations
- Download reports and plots

### Using Command Line Interface

For automated runs or research experiments:

```bash
python Livrable_3_2_WIAME_EL_HAFID.py
```

This runs default experiments:
- Grid size convergence analysis (3×3 to 9×9)
- Gamma sensitivity analysis (γ = 0.0 to 0.99)

## 🎮 How the App Works

### Core Concept

The GridWorld is a 2D grid where an intelligent agent learns to navigate from the **start position (top-left)** to a **goal position** while avoiding **obstacles** and other **competing agents**. The agent uses **Q-Learning**, a reinforcement learning algorithm, to learn the optimal policy through trial and error.

### Learning Process

1. **Initialization**: 
   - Agent starts at position [0, 0] (top-left corner)
   - Q-table initialized with zeros (no knowledge)
   - Episode counter set to 0

2. **Episode Execution**:
   - Agent observes current state (its position)
   - Chooses action using **epsilon-greedy** strategy:
     - With probability ε: random action (exploration)
     - With probability 1-ε: best known action (exploitation)
   - Executes action and observes:
     - New state (new position)
     - Reward received
     - Whether episode terminated
   
3. **Q-Value Update** (Bellman Equation):
   ```
   Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
   ```
   Where:
   - `α` (alpha) = Learning rate (how much to update)
   - `γ` (gamma) = Discount factor (importance of future rewards)
   - `r` = Immediate reward received
   - `Q(s,a)` = Current Q-value estimate
   - `max(Q(s',a'))` = Best Q-value for next state

4. **Convergence**:
   - Repeat episodes until Q-values stabilize
   - Episode terminates when:
     - Goal reached (success)
     - Collision with other agent (failure)
     - Max steps exceeded (timeout)

5. **Policy Extraction**:
   - After training, extract policy: for each state, choose action with highest Q-value
   - This policy represents the learned optimal behavior

### Key Features of the Implementation

#### 1. **Epsilon-Greedy Exploration with Decay**
```python
epsilon = max(0.05, 0.2 * 100 / (100 + t))
```
- Starts with high exploration (20%)
- Gradually decreases as learning progresses
- Never goes below 5% (maintains minimal exploration)
- `t` = step count within episode

#### 2. **Dynamic Environment Support**
- **Static entities**: Stay in fixed positions
- **Dynamic entities**: Move randomly each step
- Dynamic obstacles/goals/agents avoid collisions with each other
- Creates non-stationary environment (harder to learn)

#### 3. **Multi-Goal Support**
- Agent succeeds by reaching ANY goal
- Multiple goals provide alternative paths
- Useful for redundancy and faster learning

#### 4. **Collision Avoidance for Dynamic Entities**
- Dynamic entities use `_random_move_safe()` function
- They avoid moving onto:
  - Agent's current position
  - Other dynamic entities
  - Obstacles (for goals and other agents)
- If no safe move available, entity stays in place

#### 5. **Episode Termination Conditions**
Episodes end when:
- **Goal reached**: Agent reaches any goal position → Reward +1.0
- **Agent collision**: Agent collides with another agent → Reward -1.0
- **Max steps exceeded**: Episode timeout → Last reward applies
- **Never from obstacles/walls**: These cause penalties but episode continues

#### 6. **Real-Time Validation**
The app validates:
- No position overlaps between entities
- All positions within grid bounds
- Agent start position [0,0] not blocked
- Goal positions not blocked by obstacles
- Displays warnings for conflicts before training

## 💰 Reward Structure

The reward system is designed to encourage efficient goal-reaching behavior:

### Primary Rewards

| Event | Reward | Episode Terminates? | Description |
|-------|--------|---------------------|-------------|
| **Reach Goal** | **+1.0** | ✅ Yes | Agent successfully reaches any goal position |
| **Collide with Other Agent** | **-1.0** | ✅ Yes | Agent moves onto position occupied by another agent |
| **Bump into Obstacle** | **-0.1** | ❌ No | Agent tries to move into obstacle, stays in place |
| **Bump into Wall** | **-0.1** | ❌ No | Agent tries to move outside grid, stays in place |
| **Normal Move** | **0.0** | ❌ No | Agent successfully moves to empty cell |

### Optional: Reward Shaping

When **reward shaping** is enabled:
- **Additional penalty per step**: **-0.01**
- Applied to ALL moves (including successful moves)
- Encourages agent to find shorter paths
- Total reward for episode = Primary rewards + (steps × -0.01)

**Example without reward shaping:**
```
Episode: Start [0,0] → Move right → Move down → Move down → Reach goal [2,2]
Rewards: 0.0 + 0.0 + 0.0 + 1.0 = +1.0
```

**Example with reward shaping:**
```
Episode: Start [0,0] → Move right → Move down → Move down → Reach goal [2,2]
Rewards: (0.0-0.01) + (0.0-0.01) + (0.0-0.01) + (1.0-0.01) = +0.96
Alternative longer path (5 steps): +0.95
Reward difference encourages shorter path!
```

### Reward Philosophy

1. **Sparse Rewards**: Only goal provides significant positive reward
   - Encourages exploration
   - Agent must discover goal through trial and error

2. **Negative Feedback**: Small penalties for bumps
   - Discourages wall-hugging behavior
   - Teaches spatial awareness
   - Doesn't heavily punish exploration

3. **Terminal Penalties**: Large penalty for agent collisions
   - Teaches collision avoidance in multi-agent scenarios
   - Prevents reckless behavior

4. **Reward Shaping**: Optional efficiency incentive
   - Speeds up learning in some cases
   - Can help break symmetries in maze-like environments
   - May not always improve learning (test both!)

### Impact on Learning

**High positive rewards (+1.0 for goal):**
- Strong incentive to reach goal
- Agent quickly learns goal is valuable
- May lead to suboptimal paths initially (any path to goal is good)

**Small negative rewards (-0.1 for bumps):**
- Mild discouragement
- Doesn't overly punish exploration
- Agent still willing to try new paths

**Reward shaping effect:**
- Without: Agent learns to reach goal (any path)
- With: Agent learns to reach goal efficiently (short path)
- Trade-off: May slow convergence but better final policy

## 📊 Streamlit Interface Guide

### Left Sidebar: Configuration Panel

#### 1. 🌍 Environment Configuration

**Grid Size**: Set rows and columns (2-20)
- Determines state space size (rows × cols states)
- Larger grids = exponentially harder problem
- Start with 5×5 for experimentation

**Goals** (Required: minimum 1): 
- **Number of goals**: 1-10
- **Position options**:
  - ✅ **Default**: Automatically places goal at bottom-right [rows-1, cols-1]
  - 🎯 **Custom**: Manually specify row and column for each goal
- **Dynamic option**: ☑️ Make goals move randomly each step
  - Creates non-stationary environment
  - Significantly increases difficulty
  - Agent must learn to track moving target

**Obstacles** (Optional: 0-20):
- **Number of obstacles**: Control environment complexity
- **Position specification**: Set row and column for each obstacle
- **Dynamic option**: ☑️ Make obstacles move randomly
  - Obstacles avoid agent and other entities
  - Creates unpredictable environment
  - Tests agent's adaptability
- ⚠️ **Position validation**: App warns if obstacles conflict with other entities

**Other Agents** (Optional: 0-10):
- **Number of competing agents**: Add multi-agent complexity
- **Position specification**: Set initial positions
- **Dynamic option**: ☑️ Make agents move randomly
  - Agents move independently
  - Collision with main agent terminates episode (-1 reward)
- **Use cases**: 
  - Collision avoidance training
  - Competitive scenarios
  - Testing robustness

#### 2. 🎓 Training Configuration

**Episodes**: Number of training iterations (10-10,000)
- One episode = start to termination (goal/collision/timeout)
- More episodes = better convergence (to a point)
- Recommendations:
  - Quick test: 50-100 episodes
  - Small grids (3×3-5×5): 200-300 episodes
  - Medium grids (7×7): 500 episodes
  - Large grids (9×9+): 1000+ episodes
  - Dynamic environments: +50% episodes

**Hyperparameters**:

**Learning Rate (Alpha, α)**: 0.0-1.0 (default: 0.1)
- Controls update magnitude
- Higher α = faster learning but less stable
- Lower α = slower but more stable
- Formula influence: `Q ← Q + α[target - Q]`
- Recommendations:
  - Conservative: 0.05
  - Standard: 0.1 ✅
  - Aggressive: 0.2-0.3

**Discount Factor (Gamma, γ)**: 0.0-1.0 (default: 0.9)
- Controls importance of future rewards
- Higher γ = more long-term planning
- Lower γ = more myopic (immediate rewards)
- Formula influence: `target = r + γ·max(Q(s',a'))`
- Recommendations:
  - Short episodes: 0.8
  - Standard: 0.9 ✅
  - Long-term planning: 0.95-0.99

**Exploration Rate (Epsilon, ε)**: 0.0-1.0 (default: 0.1)
- Initial exploration probability
- **Note**: Implementation uses epsilon decay:
  - Starts at ε = 0.2 (20% random)
  - Decays to minimum 0.05 (5% random)
  - Formula: `ε = max(0.05, 0.2×100/(100+t))`
- User setting affects initial exploration strategy

**Use Default Hyperparameters**: ☑️ Quick start with proven values
- Unchecking allows manual tuning via sliders

**Max Steps per Episode**:
- **Default formula**: `max(100, rows × cols × 4)`
- Prevents infinite loops
- Should be enough for agent to reach goal
- Too low: agent times out before reaching goal
- Too high: wastes computation on failed episodes

**Reward Shaping**: ☑️ Enable -0.01 penalty per step
- Encourages efficiency
- Helps in symmetric environments
- May slow initial convergence
- Test both options to see what works better

#### 3. 💾 Output Options

**Save Trained Model**: ☑️ Export Q-table and policy
- Saves two files:
  - `model_[rows]x[cols]_ep[episodes]_q.npy` (Q-table)
  - `model_[rows]x[cols]_ep[episodes]_policy.npy` (Policy)
- Can reload and reuse trained models
- Useful for production deployment

**Export Training GIF**: ☑️ Create animated visualization
- **GIF Episodes**: How many training episodes to include (1 to total)
  - More episodes = longer GIF, bigger file
  - Recommendation: 30-50 episodes
- **GIF FPS**: Frames per second (1-30)
  - Higher = faster playback
  - Lower = easier to observe behavior
  - Recommendation: 5-10 FPS

**GIF Color Customization**: 🎨 Personalize visualization
- **Main Agent Color**: Default #00FF00 (bright green)
- **Goal Color**: Default #FF0000 (red)
- **Obstacle Color**: Default #000000 (black)
- **Other Agents Color**: Default #009600 (dark green)
- Click color boxes to open color picker
- Colors are hex codes (e.g., #FF5733)

**Generate Training Report**: ☑️ Create comprehensive text report
- Includes:
  - Complete configuration summary
  - Training statistics
  - Performance metrics
  - Convergence analysis
  - Interpretation guidance
- Saves as timestamped .txt file
- Downloadable from interface

#### 4. 📊 Plot Options

**Basic Plots**:
- ☑️ **Episode Rewards**: Reward trend over episodes
- ☑️ **Q-Delta Convergence**: Q-table stability (log scale)
- ☑️ **Combined Plot**: Both metrics on same timeline

**Advanced Analysis** (Computationally intensive):

**Grid Size Analysis**: ☑️ Compare convergence across grid dimensions
- Tests multiple grid sizes (3×3, 5×5, 7×7, 9×9)
- Runs 3 independent trials per size (statistical reliability)
- Measures episodes to convergence
- Generates plot with error bars
- **Duration**: 2-5 minutes depending on complexity
- **Use case**: Understand scaling behavior

**Gamma Sensitivity**: ☑️ Analyze discount factor impact
- Tests gamma values: 0.0, 0.3, 0.6, 0.9, 0.99
- Measures convergence speed AND reward quality
- Generates dual-metric plot
- **Option**: ☑️ Save models for each gamma
  - Creates 5 separate model files
  - Useful for comparison studies
- **Duration**: 3-5 minutes
- **Use case**: Find optimal gamma for your environment

### Main Area: Configuration Summary & Results

**Before Training**:
- Shows configuration summary
- Lists output file locations
- Displays any validation warnings

**During Training**:
- Progress bar (0-100%)
- Status text (e.g., "Training agent...", "Generating plots...")
- Real-time updates

**After Training**:
- **Training Metrics Dashboard**: 
  - Episodes completed
  - Average reward
  - Goal reached count
  - Success rate percentage
  - Training time
  - Best reward achieved
  - Final Q-delta value

- **Visualization Section**:
  - All generated plots with detailed interpretations
  - Each plot has "What it shows" and "How to interpret" guides
  - Grid analysis and gamma sensitivity results (if enabled)

- **Training Animation** (if GIF enabled):
  - Color legend showing entity meanings
  - Embedded GIF player
  - Watch agent learn in real-time

- **Training Report** (if enabled):
  - Expandable text view
  - Download button for .txt file
  - Complete training documentation

### Workflow Example

**Typical Usage Flow**:
1. Open app: `streamlit run app.py`
2. Configure 5×5 grid with goal at [4,4]
3. Add 2 obstacles at [1,2] and [2,3]
4. Keep default hyperparameters
5. Set 200 episodes
6. Enable all outputs (model, GIF, report, plots)
7. Click "🚀 Start Training"
8. Wait 30-60 seconds
9. Review results:
   - Check success rate (aim for >80%)
   - View convergence plots
   - Watch training GIF
   - Download report
10. Experiment with different configurations!

## 📦 Output Files

All outputs are organized in the `output/` folder:

### Models (`output/models/`)

```
model_[rows]x[cols]_ep[episodes]_q.npy        # Q-table (numpy array)
model_[rows]x[cols]_ep[episodes]_policy.npy   # Policy (numpy array)
model_gamma_[value]_q.npy                     # Q-table for specific gamma
model_gamma_[value]_policy.npy                # Policy for specific gamma
```

**Q-table format**: Shape (rows, cols, 4) - Q-values for each state-action pair
**Policy format**: Shape (rows, cols) - Best action for each state

### Plots (`output/plots/`)

```
rewards_[rows]x[cols].png              # Episode rewards over time
qdeltas_[rows]x[cols].png              # Q-table convergence (log scale)
combined_[rows]x[cols].png             # Both metrics together
convergence_vs_grid.png                # Grid size analysis results
gamma_sensitivity_conv_vs_reward.png   # Gamma dual-metric analysis
gamma_sensitivity_conv.png             # Gamma convergence detail
```

All plots are high-resolution PNG (150 DPI) suitable for reports/papers.

### GIFs (`output/gifs/`)

```
training_[rows]x[cols]_ep[episodes].gif  # Animated training visualization
```

**GIF specifications**:
- Customizable colors for all entities
- One frame per agent step
- Episode boundaries marked with extra frame
- Configurable FPS (frames per second)
- File size scales with episodes and grid size

### Reports (`output/reports/`)

```
report_[timestamp].txt  # Example: report_20241207_143052.txt
```

**Report contents** (70-line text file):
- Environment configuration
- Training hyperparameters
- Reward structure explanation
- Performance statistics
- Convergence metrics
- Learning progress analysis
- Interpretation guidelines

## 🎨 Understanding the Visualizations

### Visual Elements in GIFs

- **Green Circle** (customizable): Main learning agent
  - Starts at top-left [0,0]
  - Moves based on learned policy
  - Shape: Ellipse (easier to see direction)

- **Red Square** (customizable): Goal position
  - Target destination
  - Reward +1.0 when reached
  - Can be multiple goals

- **Black Square** (customizable): Obstacle
  - Blocks movement
  - Penalty -0.1 on bump
  - Agent bounces back

- **Dark Green Square** (customizable): Other competing agents
  - Move independently (if dynamic)
  - Collision causes -1 reward and episode termination
  - Smaller than main agent (85% cell size)

- **White Grid**: Navigable space
  - Agent can move freely
  - No penalty for normal moves
  - Grid lines for clarity

### Plot Interpretations

#### 1. Episode Rewards Plot
**What it shows:** Cumulative reward for each training episode.

**X-axis**: Episode number (1 to max_episodes)
**Y-axis**: Total reward for that episode

**How to interpret:**
- **Upward trend** → Agent is learning and improving
- **Values near +1.0** → Agent reaches goal efficiently with minimal bumps
- **Values around 0.0** → Agent reaches goal but hits some obstacles
- **Negative values** → Agent hits many obstacles or doesn't reach goal
- **Stabilization at high values** → Agent has learned consistent strategy
- **High variance** → Environment is stochastic or agent still exploring

**Phases you might see**:
1. **Random phase** (early): Rewards around -0.5 to 0.0 (exploration)
2. **Discovery phase** (middle): Rewards improving, high variance
3. **Optimization phase** (late): Rewards stable near +1.0

#### 2. Q-Delta Convergence Plot
**What it shows:** Maximum change in Q-values between episodes (log scale).

**X-axis**: Episode number
**Y-axis**: max|Q_new - Q_old| (logarithmic scale)

**How to interpret:**
- **Decreasing trend** → Q-table is stabilizing, agent is converging
- **Delta < 10⁻⁴** → Agent has learned near-optimal policy (converged!)
- **Delta > 10⁻²** → Still learning significantly
- **Flat line at low delta** → No more learning occurring (converged)
- **Oscillations** → Agent still exploring or environment is too stochastic

**Convergence indicators**:
- **Excellent**: Delta < 10⁻⁴ within 70% of episodes
- **Good**: Delta < 10⁻³ by end of training
- **Needs more training**: Delta > 10⁻³ at end

#### 3. Combined Analysis Plot
**What it shows:** Rewards and Q-convergence on the same timeline (dual y-axis).

**Left Y-axis (blue)**: Episode rewards
**Right Y-axis (orange)**: Max Q delta (log scale)
**X-axis**: Episode number

**How to interpret:**
- **Both improving together** → Healthy learning progress ✅
- **Rewards plateau, Q still changing** → Fine-tuning without performance gain
- **Q converged, rewards unstable** → Stochastic environment or continued exploration
- **Neither improving** → Problem with hyperparameters or environment is too hard

**Ideal pattern**:
- Rewards increase and stabilize
- Q-delta decreases smoothly
- Both reach stable state simultaneously

#### 4. Grid Size Analysis Plot
**What it shows:** Episodes needed for convergence vs grid dimensions.

**X-axis**: Grid size (3×3, 5×5, 7×7, 9×9)
**Y-axis**: Convergence episode (mean ± std over 3 runs)
**Error bars**: Standard deviation across runs

**How to interpret:**

**Growth patterns**:
- **Linear** (convergence ∝ grid_cells): Q-learning scales well ✅
- **Quadratic** (convergence ∝ grid_cells²): Acceptable scaling
- **Exponential**: Poor scaling, consider DQN for larger grids

**Error bars**:
- **Small bars**: Consistent convergence (reliable)
- **Large bars**: High variance (environment may be too random)

**Practical implications**:
- Predict training time for larger environments
- Assess Q-learning suitability
- Decide if algorithm change needed (e.g., Deep Q-Learning)
- Plan computational budget for production

**Example interpretation**:
```
3×3 → 50±5 episodes    |  Linear growth
5×5 → 120±10 episodes  |  → Q-learning is good choice
7×7 → 220±15 episodes  |  → Can scale to 10×10
9×9 → 350±30 episodes  |
```

#### 5. Gamma Sensitivity Plot
**What it shows:** How discount factor (γ) affects learning and performance.

**X-axis**: Gamma values (0.0, 0.3, 0.6, 0.9, 0.99)
**Left Y-axis (blue)**: Episodes to converge (lower = faster)
**Right Y-axis (orange)**: Average reward (higher = better)

**Gamma meanings**:
- **γ = 0.0**: Only immediate rewards matter (myopic agent)
- **γ = 0.5**: Balance short/long term
- **γ = 0.9**: Strong future planning (standard)
- **γ = 0.99**: Maximum long-term optimization

**How to interpret**:

**Convergence speed (blue curve)**:
- Low γ → Fast convergence (simple policies)
- High γ → Slower convergence (complex planning)
- **Trade-off**: Speed vs quality

**Reward quality (orange curve)**:
- Low γ → Lower rewards (short-sighted decisions)
- Medium γ → Good rewards
- High γ → Best rewards (if converges)
- **Sweet spot**: Where curve plateaus

**Decision-making**:
1. Find where reward curve plateaus
2. Choose lowest γ at plateau (faster training)
3. Example: If γ=0.9 gives 0.85 reward and γ=0.95 gives 0.86 reward
   → Choose γ=0.9 (not worth 50% more training for 1% reward gain)

**Typical patterns**:
- **Simple environment**: Sweet spot at γ=0.8-0.9
- **Complex environment**: Sweet spot at γ=0.95-0.99
- **Maze-like**: Higher γ needed for long-term planning

## 📄 Training Reports

Each report is a comprehensive text document with multiple sections:

### 1. Header Section
```
======================================================================
GridWorld Q-Learning Training Report
======================================================================
Generated: 2024-12-07 14:30:52
```

### 2. Environment Configuration
- Grid dimensions and total cells
- Goal count, positions, and dynamics
- Obstacle count, positions, and dynamics
- Other agent configuration
- Max steps per episode

### 3. Training Configuration
- Episode count
- Learning rate (alpha)
- Discount factor (gamma)
- Exploration rate (epsilon)
- Reward shaping status

### 4. Reward Structure
- Complete reward table
- Termination conditions
- Behavioral implications

### 5. Training Results
- Total training time
- Episodes completed
- Goal reached count and percentage
- Episode reward statistics:
  - Average reward
  - Final episode reward
  - Best/worst episode rewards
  - Standard deviation
- Learning progress analysis (first 10 vs last 10 episodes)

### 6. Q-Table Convergence
- Initial and final Q-delta values
- Convergence status (yes/no)
- Convergence episode (if applicable)

### 7. Interpretation Guide
- How to read reward metrics
- Success rate benchmarks (>80% excellent, 50-80% good, <50% needs work)
- Q-delta convergence thresholds
- Recommendations for improvement

**Example report snippet**:
```
TRAINING RESULTS
----------------------------------------------------------------------
Training Time: 23.45 seconds
Episodes Completed: 200
Goal Reached: 178 times (89.0%)

Episode Rewards:
  Average: 0.8234
  Final Episode: 0.9500
  Best Episode: 1.0000 (Episode 156)
  Worst Episode: -0.3200 (Episode 12)
  Standard Deviation: 0.2156

Learning Progress:
  First 10 Episodes Avg: 0.2340
  Last 10 Episodes Avg: 0.9280
  Improvement: +0.6940 (+296.6%)
```

## 🔧 Advanced Usage

### Custom Environment Configuration

```python
from Livrable_3_2_WIAME_EL_HAFID import GridWorldEnv, QLearningAgent

# Create custom environment
env = GridWorldEnv(
    rows=7, cols=7,
    n_goals=2,
    goals_pos=[[6,6], [3,3]],
    goals_dynamic=False,
    n_obstacles=3,
    obstacles_pos=[[2,2], [3,4], [5,1]],
    obstacles_dynamic=True,
    n_other_agents=1,
    other_agents_pos=[[4,4]],
    other_agents_dynamic=True,
    render=False,
    max_steps=200,
    seed=123
)

# Train agent
agent = QLearningAgent(env, episodes=500, alpha=0.1, gamma=0.9)
rewards, deltas, lengths = agent.train_track()

# Save model
agent.save_model('output/models/my_custom_model')
```

### Loading Saved Models

```python
from Livrable_3_2_WIAME_EL_HAFID import QLearningAgent

# Load saved model
model_data = QLearningAgent.load_model('output/models/my_custom_model')
Q = model_data['Q']
policy = model_data['policy']

# Use loaded policy for inference
def get_action(state):
    return policy[state[0], state[1]]
```

### Export Training GIF with Custom Colors

```python
from Livrable_3_2_WIAME_EL_HAFID import export_training_gif_custom_colors

custom_colors = {
    'agent': '#00FF00',      # Bright green
    'goal': '#FF0000',       # Red
    'obstacle': '#000000',   # Black
    'other_agent': '#0000FF' # Blue
}

export_training_gif_custom_colors(
    env, agent,
    episodes=50,
    out_path='custom_training.gif',
    fps=10,
    colors=custom_colors
)
```

### Batch Experimentation

```python
import numpy as np
from Livrable_3_2_WIAME_EL_HAFID import GridWorldEnv, QLearningAgent

# Test different gamma values
gammas = [0.0, 0.3, 0.6, 0.9, 0.99]
results = {}

for gamma in gammas:
    env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], 
                       n_obstacles=2, obstacles_pos=[[1,2],[2,3]], render=False)
    agent = QLearningAgent(env, episodes=200, gamma=gamma)
    rewards, deltas, lengths = agent.train_track()
    
    results[gamma] = {
        'avg_reward': np.mean(rewards),
        'convergence_ep': len(deltas),
        'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards)
    }
    
    agent.save_model(f'output/models/model_gamma_{gamma:.2f}')
    env.close()
    print(f"Gamma {gamma}: Avg Reward = {results[gamma]['avg_reward']:.3f}, "
          f"Success = {results[gamma]['success_rate']*100:.1f}%")

# Find best gamma
best_gamma = max(results, key=lambda g: results[g]['avg_reward'])
print(f"\nBest gamma: {best_gamma} with avg reward {results[best_gamma]['avg_reward']:.3f}")
```

## ⚡ Performance Benchmarks

### Convergence Times (Approximate)

| Grid Size | State Space | Episodes to Converge | Training Time | Memory Usage |
|-----------|-------------|----------------------|---------------|--------------|
| 3×3 | 9 states | 30-60 | ~5 seconds | 10 MB |
| 5×5 | 25 states | 80-150 | ~15 seconds | 25 MB |
| 7×7 | 49 states | 150-300 | ~45 seconds | 50 MB |
| 9×9 | 81 states | 300-600 | ~2 minutes | 100 MB |
| 12×12 | 144 states | 600-1200 | ~5 minutes | 200 MB |
| 15×15 | 225 states | 1000-2000 | ~10 minutes | 300 MB |

*Times measured on standard laptop (Intel i5, 8GB RAM, no GPU)*

### Factors Affecting Performance

**Speed**:
- Grid size (exponential impact)
- Number of episodes
- Dynamic entities (adds computation per step)
- Rendering enabled (10-50× slower)
- GIF export (adds overhead)

**Memory**:
- Q-table size: `rows × cols × 4 × 8 bytes`
- Example: 10×10 grid = 10 × 10 × 4 × 8 = 3,200 bytes (negligible)
- Training data: rewards, deltas, lengths arrays
- GIF frames: Most memory-intensive (if enabled)

### Scaling Recommendations

**Q-Learning (Tabular)**:
- ✅ **Excellent**: Grids < 10×10 (< 100 states)
- ⚠️ **Feasible**: Grids 10×10 to 15×15 (100-225 states)
- ❌ **Not Recommended**: Grids > 15×15 (> 225 states)

**For Larger Grids**:
- Use Deep Q-Learning (DQN) with neural networks
- Implement function approximation
- Consider hierarchical reinforcement learning
- Use state abstraction techniques

### Optimization Tips

**For Faster Training**:
```python
# Disable rendering
env = GridWorldEnv(..., render=False)

# Reduce max_steps if possible
env = GridWorldEnv(..., max_steps=100)

# Skip GIF export during development
# Only export final trained model

# Use smaller grid for prototyping
# Scale up once hyperparameters tuned
```

**For Better Policies**:
```python
# Increase training episodes
agent = QLearningAgent(env, episodes=1000)

# Use higher gamma for long-term planning
agent = QLearningAgent(env, gamma=0.95)

# Enable reward shaping for efficiency
agent = QLearningAgent(env, reward_shaping=True)

# Reduce exploration after some learning
# (implementation already uses epsilon decay)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: Pygame Display Error
```
Warning: rendering disabled; could not initialize pygame display
```

**Cause**: No display server available (headless system, SSH without X11)

**Solution**: This is expected behavior
- Rendering is optional
- All functionality works without display
- Use GIF export for visualization
- Training proceeds normally

**Alternative**: If you need live rendering
- On Linux: Install X server (`sudo apt install xorg`)
- On WSL: Use VcXsrv or similar X server
- On remote server: Use X11 forwarding (`ssh -X`)

#### Issue: Port Already in Use
```
Error: Address already in use
```

**Cause**: Another Streamlit instance running on port 8501

**Solution**:
```bash
# Option 1: Use different port
streamlit run app.py --server.port 8502

# Option 2: Kill existing process
# Linux/Mac
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

#### Issue: Position Conflicts Warning
```
⚠️ Goal 1 at (2, 3) conflicts with another entity!
```

**Cause**: Multiple entities placed at same grid cell

**Solution**:
- Check entity positions in sidebar
- Ensure no overlaps:
  - Agent starts at [0, 0]
  - Each goal has unique position
  - Obstacles don't overlap goals/agent start
  - Other agents have unique positions
- App validates automatically and shows warnings

#### Issue: Module Import Errors
```
ModuleNotFoundError: No module named 'gymnasium'
```

**Cause**: Dependencies not installed

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep gymnasium
```

#### Issue: Agent Not Learning
**Symptoms**: Rewards stay negative or don't improve

**Possible Causes & Solutions**:

1. **Goal is unreachable** (blocked by obstacles)
   - Solution: Verify path exists from [0,0] to goal
   - Check obstacle placement

2. **Not enough episodes**
   - Solution: Increase episodes (try 2-3× current)
   - Check Q-delta plot for convergence

3. **Max steps too low**
   - Solution: Increase max_steps or use default
   - Agent needs time to reach goal

4. **Too much exploration**
   - Solution: Training uses epsilon decay automatically
   - Check if using very high initial epsilon

5. **Environment too stochastic** (all dynamic)
   - Solution: Start with static entities
   - Add dynamics gradually

6. **Hyperparameters not suitable**
   - Solution: Try alpha=0.2, gamma=0.95
   - Run gamma sensitivity analysis

**Debugging Steps**:
```python
# 1. Test simple environment first
env = GridWorldEnv(rows=3, cols=3, n_goals=1, goals_pos=[[2,2]], 
                   n_obstacles=0, render=False)
agent = QLearningAgent(env, episodes=100)
rewards, _, _ = agent.train_track()
print(f"Avg reward: {np.mean(rewards)}")  # Should be > 0.5

# 2. Gradually increase complexity
# 3. Check each addition's impact
```

#### Issue: Training Too Slow
**Symptoms**: Takes minutes for small grids

**Solutions**:
```python
# Ensure render=False
env = GridWorldEnv(..., render=False)

# Reduce episodes for testing
agent = QLearningAgent(env, episodes=50)

# Skip GIF export during development

# Use smaller grid for prototyping
env = GridWorldEnv(rows=3, cols=3, ...)

# Check system resources
# CPU usage should be high during training
```

#### Issue: Empty or Single-Point Plots

**Cause**: Training completed too quickly or data not captured

**Solution**:
- Ensure episodes > 1 (ideally 50+)
- Check console for error messages
- Verify `train_track()` returned data
- Look for premature early stopping

#### Issue: Out of Memory
```
MemoryError: Unable to allocate array
```

**Cause**: Grid too large or too many episodes

**Solution**:
- Reduce grid size (< 15×15 recommended)
- Lower episode count
- Close other applications
- Don't create multiple large environments simultaneously
- For very large grids, use Deep Q-Learning instead

#### Issue: GIF Not Generated

**Possible Causes**:
1. **imageio not installed**: `pip install imageio`
2. **Insufficient disk space**: Check available space
3. **Permission error**: Check write permissions on output/gifs/
4. **Too many episodes**: Reduce gif_episodes parameter

**Solution**:
```python
# Test GIF export manually
from Livrable_3_2_WIAME_EL_HAFID import export_training_gif_custom_colors

env = GridWorldEnv(rows=3, cols=3, render=False)
agent = QLearningAgent(env, episodes=10)

try:
    export_training_gif_custom_colors(
        env, agent, episodes=10, 
        out_path='test.gif', fps=5
    )
    print("GIF export successful!")
except Exception as e:
    print(f"GIF export failed: {e}")
```

#### Issue: Streamlit App Won't Start

**Error**: `streamlit: command not found`

**Solution**:
```bash
# Ensure streamlit is installed
pip install streamlit

# Verify installation
streamlit --version

# If still fails, use python -m
python -m streamlit run app.py
```

#### Issue: Browser Doesn't Auto-Open

**Solution**:
- Manually open browser to `http://localhost:8501`
- Check firewall isn't blocking port 8501
- Try different browser
- Check terminal for correct URL

## ❓ FAQ

### General Questions

**Q: What is Q-Learning?**  
A: Q-Learning is a model-free reinforcement learning algorithm that learns the value (Q-value) of taking each action in each state. The agent uses these Q-values to choose actions that maximize cumulative reward.

**Q: How does the agent learn?**  
A: Through trial and error! The agent explores the environment, receives rewards/penalties, and updates its Q-table using the Bellman equation. Over many episodes, it learns which actions lead to the goal.

**Q: What's the difference between Q-table and Policy?**  
A: 
- **Q-table**: Stores expected rewards for each state-action pair (rows × cols × 4 values)
- **Policy**: Derived from Q-table, stores best action for each state (rows × cols values)
- Policy is what you use for decision-making after training

### Environment Questions

**Q: Can I change the colors in the visualization?**  
A: Yes! In the Streamlit interface, under "Output Options", expand "GIF Color Customization" and click the color boxes to choose custom colors for all entities.

**Q: How do I make the agent learn faster?**  
A: Try these approaches:
- Increase learning rate (alpha to 0.2)
- Reduce exploration (epsilon to 0.05)
- Enable reward shaping
- Simplify environment (fewer obstacles, static entities)
- Increase episodes

**Q: Can I add more than one goal?**  
A: Yes! Set "Number of Goals" to any value 1-10 and configure each position. Agent succeeds by reaching ANY goal, which can speed up learning.

**Q: What's the difference between static and dynamic entities?**  
A: 
- **Static**: Entities stay in fixed positions (easier to learn)
- **Dynamic**: Entities move randomly each step (non-stationary environment, much harder)
- Dynamic entities avoid collisions with agent and each other

**Q: Why does my agent keep hitting obstacles?**  
A: Early in training, this is normal (exploration). If it persists:
- Increase training episodes
- Check if obstacle penalty (-0.1) is sufficient
- Try enabling reward shaping
- Verify goal is reachable

**Q: Can I have obstacles block all paths to the goal?**  
A: Technically yes, but the agent will never reach the goal. The app doesn't validate path existence, so ensure at least one path from [0,0] to goal exists.

### Training Questions

**Q: How many episodes do I need?**  
A: Depends on complexity:
- 3×3 grid, no obstacles: 50-100 episodes
- 5×5 grid, 2 obstacles: 200-300 episodes
- 7×7 grid, 3 obstacles: 500+ episodes
- Dynamic environments: Add 50-100%
- Check Q-delta plot for convergence

**Q: What are good hyperparameter values?**  
A: Defaults work well for most cases:
- Alpha (α): 0.1 (learning rate)
- Gamma (γ): 0.9 (discount factor)
- Epsilon (ε): Uses decay from 0.2 to 0.05
- For specific environments, run gamma sensitivity analysis

**Q: How do I know if training is complete?**  
A: Look for these indicators:
- Q-delta < 10⁻⁴ (converged!)
- Rewards stabilized near +1.0
- Success rate > 80%
- Plot shows flat convergence
- Little change in last 50 episodes

**Q: Should I use reward shaping?**  
A: Test both! 
- **Pros**: Encourages shorter paths, can speed learning
- **Cons**: May slow initial convergence, not always beneficial
- **When to use**: Symmetric environments, when path length matters
- **When to skip**: Simple environments, when just reaching goal matters

**Q: What does "episode terminated" mean?**  
A: Episode ends when:
- Agent reaches goal (success, reward +1)
- Agent collides with another agent (failure, reward -1)
- Max steps exceeded (timeout, last reward applies)

### Results Questions

**Q: How do I interpret the training report?**  
A: Key metrics to check:
- **Success rate**: >80% excellent, 50-80% good, <50% needs work
- **Average reward**: >0.5 good, >0.8 excellent
- **Final Q-delta**: <10⁻⁴ converged, >10⁻³ needs more training
- **Improvement**: Compare first 10 vs last 10 episodes (should show growth)

**Q: My rewards are negative. Is that bad?**  
A: Early episodes often have negative rewards (exploration). If rewards stay negative:
- Agent isn't reaching goal
- Too many obstacle collisions
- Need more training episodes
- Check if goal is reachable

**Q: What's a good success rate?**  
A: 
- **80-100%**: Excellent! Agent learned well
- **50-80%**: Good, might improve with more training
- **<50%**: Needs work - more episodes or adjust hyperparameters

**Q: Why do my plots show oscillations?**  
A: Several reasons:
- Agent still exploring (normal early on)
- Environment is stochastic (dynamic entities)
- Epsilon not decayed enough
- Learning rate too high (try lower alpha)

### Advanced Questions

**Q: Can I use this for research?**  
A: Yes! Code provided for educational and research purposes. Please cite appropriately (see [Citation](#citation) section).

**Q: How do I compare two trained agents?**  
A: 
1. Train both on same environment (use same seed)
2. Compare success rates, average rewards, convergence speed
3. Run multiple trials (at least 3) for statistical significance
4. Load and test both policies on evaluation episodes
5. Use t-test for statistical comparison

**Q: Can I load a saved model and continue training?**  
A: Current implementation doesn't support resuming. You can load Q-table and use it, but continuing training would require modifying the code.

**Q: Does this work with continuous state spaces?**  
A: No, this is discrete Q-learning (tabular). For continuous spaces, you need:
- State discretization
- Function approximation (neural networks)
- Deep Q-Learning (DQN)

**Q: Can I use this for actual robot navigation?**  
A: This is a learning tool/simulator. For real robots:
- Train in simulation first
- Consider sim-to-real transfer challenges
- Add sensor noise, actuation uncertainty
- Use more robust RL algorithms (PPO, SAC)
- Test extensively in safe environment

**Q: How does this compare to Deep Q-Learning?**  
A: 
- **Q-Learning (this)**: Fast for small state spaces, exact, simple
- **DQN**: Scales to large/continuous spaces, uses neural networks, more complex
- **Use Q-Learning when**: Grid < 15×15, discrete states, fast prototyping
- **Use DQN when**: Large state space, continuous states, complex environments

**Q: Can I add custom rewards?**  
A: Yes! Edit the `step()` method in `Livrable_3_2_WIAME_EL_HAFID.py`. Example:
```python
# In step() method, add custom reward logic
if custom_condition:
    reward += custom_bonus
```

## 💡 Tips for Best Results

### For Beginners

1. **Start with simplest environment**:
   - 5×5 grid
   - 1 goal at [4,4]
   - 2 static obstacles
   - No other agents
   - 200 episodes

2. **Use default hyperparameters**:
   - α = 0.1
   - γ = 0.9  
   - ε = 0.1 (with decay)

3. **Enable all outputs**:
   - Save model
   - Generate plots
   - Create GIF
   - Generate report

4. **Read the visualizations**:
   - Check if rewards improve
   - Verify Q-delta decreases
   - Watch GIF to understand behavior

5. **Experiment systematically**:
   - Change ONE parameter at a time
   - Compare results
   - Document findings

### For Advanced Users

1. **Profile your environment**:
   - Run grid size analysis
   - Run gamma sensitivity
   - Identify optimal parameters

2. **Systematic hyperparameter search**:
   ```python
   alphas = [0.05, 0.1, 0.2]
   gammas = [0.8, 0.9, 0.95]
   
   for a in alphas:
       for g in gammas:
           # Train and evaluate
           # Record results
   ```

3. **Use statistical validation**:
   - Run each configuration 5-10 times
   - Report mean ± std
   - Use proper statistical tests

4. **Benchmark against baselines**:
   - Random policy
   - Hand-coded policy
   - Other RL algorithms

5. **Document thoroughly**:
   - Save all configurations
   - Keep training logs
   - Track experiment history

### For Researchers

1. **Reproducibility**:
   - Always set seeds
   - Document exact versions
   - Save complete configuration
   - Use version control

2. **Experimental design**:
   - Define hypotheses clearly
   - Control variables properly
   - Use adequate sample sizes
   - Report all metrics

3. **Analysis**:
   - Statistical significance testing
   - Confidence intervals
   - Learning curves
   - Ablation studies

4. **Comparison**:
   - Fair baselines
   - Same evaluation protocol
   - Multiple random seeds
   - Standard benchmarks

5. **Publication**:
   - Clear methodology
   - Reproducible results
   - Open-source code
   - Cite appropriately

## 📚 Documentation

- **README.md** (this file): Project overview and comprehensive guide
- **USER_GUIDE.md**: 10-section detailed user manual with FAQs
- **GITHUB_SETUP.md**: Step-by-step GitHub deployment instructions
- **CONTRIBUTING.md**: Guidelines for contributors

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution**:
- 🎮 New environment features (walls, teleports, power-ups)
- 🧠 Additional RL algorithms (SARSA, DQN, Actor-Critic)
- 📊 Visualization improvements (3D plots, heatmaps)
- ⚡ Performance optimizations (numba, cython)
- 📖 Documentation enhancements
- 🐛 Bug fixes
- 🧪 Unit tests

**How to contribute**:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with clear commits
4. Test thoroughly
5. Submit pull request

## 📖 Citation

If using this project for academic purposes, please cite:

```bibtex
@software{gridworld_qlearning_2024,
  author = {Your Name},
  title = {GridWorld Q-Learning Training Platform},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/gridworld-qlearning},
  note = {Interactive reinforcement learning platform with comprehensive analysis tools}
}
```

**References**:
- Watkins, C.J.C.H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ No Liability
- ❌ No Warranty

## 🙏 Acknowledgments

- **Gymnasium** (formerly OpenAI Gym): Environment interface
- **Streamlit**: Interactive web interface framework
- **Matplotlib**: Plotting and visualization
- **Pygame**: Real-time rendering
- **NumPy**: Numerical computations
- **Pillow & imageio**: GIF generation

## 📧 Contact

For questions, suggestions, or collaboration:

<<<<<<< HEAD
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/gridworld-qlearning/issues)
- **GitHub Discussions**: [Join the discussion](https://github.com/yourusername/gridworld-qlearning/discussions)
- **Email**: your.email@example.com
=======
- **GitHub Issues**: [Create an issue](https://github.com/WE2722/gridworld-qlearning/issues)
- **GitHub Discussions**: [Join the discussion](https://github.com/WE2722/gridworld-qlearning/discussions)
- **Email**: wiame.el.hafid27@example.com
>>>>>>> c508dcdad35659ecf690dcd614a41bb5b702b04f

## 🌟 Star History

If you find this project useful:
- ⭐ Give it a star on GitHub
- 🍴 Fork it for your own experiments
- 📢 Share it with others
- 💬 Provide feedback

## 🔗 Related Projects

- [OpenAI Gym](https://github.com/openai/gym): Original RL environment toolkit
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): PyTorch RL implementations
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Scalable RL library
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Clean RL implementations

---

<<<<<<< HEAD
**Made with ❤️ for the Reinforcement Learning community**

**Start learning today**: `streamlit run app.py`# 🤖 GridWorld Q-Learning Reinforcement Learning Project

A comprehensive GridWorld environment with Q-Learning agent training, featuring an interactive Streamlit interface for easy configuration and visualization.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## ✨ Features

- **Interactive Streamlit Web Interface**: Configure all parameters through a user-friendly GUI
- **Customizable GridWorld Environment**: Configure grid size, goals, obstacles, and other agents
- **Q-Learning Implementation**: Classic reinforcement learning with epsilon-greedy exploration
- **Training Visualization**: Animated GIF export of the learning process with customizable colors
- **Model Persistence**: Save and load trained Q-tables and policies
- **Comprehensive Analysis**:
  - Grid size impact on convergence
  - Gamma (discount factor) sensitivity analysis
  - Models saved for different gamma values
- **Training Reports**: Automatic generation of detailed training reports with interpretations
- **Organized Output Structure**: All outputs saved in organized folders

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Streamlit Interface Guide](#streamlit-interface-guide)
- [Output Files](#output-files)
- [Understanding the Visualizations](#understanding-the-visualizations)
- [Training Reports](#training-reports)
- [Advanced Usage](#advanced-usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/gridworld-qlearning.git
cd gridworld-qlearning
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
gymnasium>=0.29.0
numpy>=1.24.0
pygame>=2.5.0
matplotlib>=3.7.0
imageio>=2.31.0
Pillow>=10.0.0
streamlit>=1.28.0
```

### Quick Setup Script

**Linux/macOS:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

**Windows:**
```bash
quickstart.bat
```

## 📁 Project Structure

```
gridworld-qlearning/
│
├── app.py                              # Streamlit web interface
├── Livrable_3_2_WIAME_EL_HAFID.py     # Core environment & Q-Learning
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── USER_GUIDE.md                      # Detailed user guide
├── GITHUB_SETUP.md                    # GitHub setup instructions
├── quickstart.sh                      # Setup script (Linux/macOS)
├── quickstart.bat                     # Setup script (Windows)
│
└── output/                            # Generated outputs
    ├── models/                        # Saved Q-tables and policies
    ├── plots/                         # Training visualizations
    ├── gifs/                          # Training animations
    └── reports/                       # Training reports
```

## 🎯 Quick Start

### Using Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

This will open a web browser with an interactive interface where you can:
- Configure the environment visually
- Set training parameters with sliders
- Choose which outputs to generate
- View results in real-time with detailed interpretations
- Download reports and plots

### Using Command Line Interface

For automated runs or research experiments:

```bash
python Livrable_3_2_WIAME_EL_HAFID.py
```

This runs default experiments:
- Grid size convergence analysis (3×3 to 9×9)
- Gamma sensitivity analysis (γ = 0.0 to 0.99)

## 📊 Streamlit Interface Guide

### Configuration Sections

#### 1. 🌍 Environment Configuration

**Grid Size**: Set rows and columns (2-20)

**Goals**: 
- Number of goals (1-10)
- Choose default position (bottom-right) or custom positions
- Option to make goals move randomly

**Obstacles**:
- Number of obstacles (0-20)
- Set individual positions for each obstacle
- Option to make obstacles move randomly
- ⚠️ Automatic validation prevents position conflicts

**Other Agents**:
- Number of competing agents (0-10)
- Set individual positions
- Option to make them move randomly
- Collision with other agents terminates episode with -1 reward

#### 2. 🎓 Training Configuration

**Episodes**: Number of training episodes (10-10,000)

**Hyperparameters**:
- Learning rate (alpha): 0.0-1.0 (default: 0.1)
- Discount factor (gamma): 0.0-1.0 (default: 0.9)
- Exploration rate (epsilon): 0.0-1.0 (default: 0.1)

**Max Steps**: Use default (rows × cols × 4) or set custom max steps per episode

**Reward Shaping**: Enable small penalty (-0.01) per step for efficiency

#### 3. 💾 Output Options

**Save Model**: Save Q-table and policy as .npy files

**Export GIF**: Create training animation
- Set number of episodes to include
- Set frames per second (FPS)
- **Customize Colors**: Pick colors for agent, goal, obstacles, and other agents

**Generate Report**: Create detailed training report with full analysis

#### 4. 📊 Plot Selection

Choose which plots to generate:
- ✅ Episode Rewards over time
- ✅ Q-Delta Convergence (log scale)
- ✅ Combined convergence plot

**Advanced Analysis:**
- 📐 **Grid Size Analysis**: Compare convergence across different grid sizes with error bars
- 🎯 **Gamma Sensitivity**: Analyze impact of discount factor with dual-metric plots
  - Option to save models for each gamma value

## 📦 Output Files

All outputs are organized in the `output/` folder:

### Models (`output/models/`)

```
model_[rows]x[cols]_ep[episodes]_q.npy        # Q-table
model_[rows]x[cols]_ep[episodes]_policy.npy   # Policy
model_gamma_[value]_q.npy                     # Q-table for specific gamma
model_gamma_[value]_policy.npy                # Policy for specific gamma
```

### Plots (`output/plots/`)

```
rewards_[rows]x[cols].png              # Episode rewards
qdeltas_[rows]x[cols].png              # Q-table convergence
combined_[rows]x[cols].png             # Combined plot
convergence_vs_grid.png                # Grid size analysis
gamma_sensitivity_conv_vs_reward.png   # Gamma sensitivity (dual metric)
gamma_sensitivity_conv.png             # Gamma convergence detail
```

### GIFs (`output/gifs/`)

```
training_[rows]x[cols]_ep[episodes].gif  # Training animation with custom colors
```

### Reports (`output/reports/`)

```
report_[timestamp].txt  # Comprehensive training report
```

## 🎨 Understanding the Visualizations

### Visual Elements in GIFs

- **Green Circle** (customizable): Main agent (learning)
- **Red Square** (customizable): Goal position (reward +1)
- **Black Square** (customizable): Obstacle (penalty -0.1 on bump)
- **Dark Green Square** (customizable): Other agents (penalty -1 on collision)
- **White Grid**: Navigable space

### Plot Interpretations

#### Episode Rewards Plot
**What it shows:** Cumulative reward for each training episode.

**How to interpret:**
- **Upward trend** → Agent is learning and improving
- **Values near +1.0** → Agent reaches goal efficiently
- **Negative values** → Agent hits obstacles or fails to reach goal
- **Stabilization** → Agent has learned a consistent strategy

#### Q-Delta Convergence Plot
**What it shows:** Maximum change in Q-values between episodes (log scale).

**How to interpret:**
- **Decreasing trend** → Q-table is stabilizing, agent is converging
- **Delta < 10⁻⁴** → Agent has learned near-optimal policy
- **Flat line** → No more learning occurring
- **Oscillations** → Agent still exploring or environment is too complex

#### Combined Analysis Plot
**What it shows:** Rewards and Q-convergence on the same timeline.

**How to interpret:**
- **Both improving together** → Healthy learning progress
- **Rewards plateau but Q still changing** → Fine-tuning strategy
- **Rewards unstable but Q converged** → Stochastic environment

#### Grid Size Analysis Plot
**What it shows:** Episodes needed for convergence vs grid dimensions.

**How to interpret:**
- **Linear growth** → Q-learning scales well
- **Exponential growth** → Consider Deep Q-Learning for larger grids
- **Error bars** → Show consistency across runs (smaller = more reliable)

**Practical implications:**
- Predict training time for larger environments
- Assess scalability of Q-learning
- Decide if function approximation is needed

#### Gamma Sensitivity Plot
**What it shows:** How discount factor affects learning and performance.

**Two metrics:**
- **Blue (left axis)**: Episodes to converge (lower = faster)
- **Orange (right axis)**: Average reward (higher = better)

**How to interpret:**
- **Low γ (0.0-0.3)**: Fast convergence, lower rewards (short-sighted)
- **Optimal γ (0.9)**: Good balance of speed and performance
- **High γ (0.99)**: Best rewards but slower convergence (long-term planning)

**Use this to:** Find the sweet spot between training speed and policy quality

## 📄 Training Reports

Each report includes:

### Environment Configuration
- Grid dimensions and total cells
- Goal positions and dynamics
- Obstacle configuration
- Other agents setup
- Max steps per episode

### Training Configuration
- Number of episodes
- Hyperparameters (α, γ, ε)
- Reward shaping enabled/disabled

### Reward Structure
- Reach goal: +1.0 (terminates)
- Collide with agent: -1.0 (terminates)
- Bump obstacle/wall: -0.1 (continues)
- Normal move: 0.0 (continues)

### Training Results
- Training time
- Goal reached count and percentage
- Average/best/worst episode rewards
- Q-table convergence metrics
- Learning progress (first 10 vs last 10 episodes)

### Interpretation Guide
- Performance benchmarks
- Success rate categories
- Convergence indicators

## 🔧 Advanced Usage

### Custom Environment Configuration

```python
from Livrable_3_2_WIAME_EL_HAFID import GridWorldEnv, QLearningAgent

# Create custom environment
env = GridWorldEnv(
    rows=7, cols=7,
    n_goals=2,
    goals_pos=[[6,6], [3,3]],
    goals_dynamic=False,
    n_obstacles=3,
    obstacles_pos=[[2,2], [3,4], [5,1]],
    obstacles_dynamic=True,
    n_other_agents=1,
    other_agents_pos=[[4,4]],
    other_agents_dynamic=True,
    render=False,
    max_steps=200,
    seed=123
)

# Train agent
agent = QLearningAgent(env, episodes=500, alpha=0.1, gamma=0.9)
rewards, deltas, lengths = agent.train_track()

# Save model
agent.save_model('output/models/my_custom_model')
```

### Loading Saved Models

```python
from Livrable_3_2_WIAME_EL_HAFID import QLearningAgent

# Load saved model
model_data = QLearningAgent.load_model('output/models/my_custom_model')
Q = model_data['Q']
policy = model_data['policy']

# Use loaded policy for inference
def get_action(state):
    return policy[state[0], state[1]]
```

### Export Training GIF with Custom Colors

```python
from Livrable_3_2_WIAME_EL_HAFID import export_training_gif_custom_colors

custom_colors = {
    'agent': '#00FF00',      # Bright green
    'goal': '#FF0000',       # Red
    'obstacle': '#000000',   # Black
    'other_agent': '#0000FF' # Blue
}

export_training_gif_custom_colors(
    env, agent,
    episodes=50,
    out_path='custom_training.gif',
    fps=10,
    colors=custom_colors
)
```

### Batch Experimentation

```python
# Test different gamma values
gammas = [0.0, 0.3, 0.6, 0.9, 0.99]
results = {}

for gamma in gammas:
    env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], 
                       n_obstacles=2, obstacles_pos=[[1,2],[2,3]], render=False)
    agent = QLearningAgent(env, episodes=200, gamma=gamma)
    rewards, deltas, lengths = agent.train_track()
    
    results[gamma] = {
        'avg_reward': np.mean(rewards),
        'convergence_ep': len(deltas)
    }
    
    agent.save_model(f'output/models/model_gamma_{gamma:.2f}')
    env.close()
    print(f"Gamma {gamma}: Avg Reward = {results[gamma]['avg_reward']:.3f}")
```

## ⚡ Performance Benchmarks

### Convergence Times (Approximate)

| Grid Size | Episodes to Converge | Training Time |
|-----------|---------------------|---------------|
| 3×3 | 30-60 | ~5 seconds |
| 5×5 | 80-150 | ~15 seconds |
| 7×7 | 150-300 | ~45 seconds |
| 9×9 | 300-600 | ~2 minutes |

*Times measured on standard laptop (Intel i5, 8GB RAM)*

### Memory Usage

- Small grids (3×3 to 5×5): <50 MB
- Medium grids (7×7): ~100 MB
- Large grids (9×9+): 150-300 MB

### Scaling Recommendations

- **Grids < 10×10**: Q-learning works well
- **Grids 10×10 to 15×15**: Q-learning feasible but slow
- **Grids > 15×15**: Consider Deep Q-Learning (DQN)

## 🐛 Troubleshooting

### Streamlit Issues

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Browser doesn't open automatically:**
```bash
# Manually open: http://localhost:8501
```

### Pygame Display Issues

If you see "rendering disabled" warning:
- This is expected for headless systems
- Use GIF export instead of live visualization
- Rendering is optional; all functionality works without it

### Module Import Errors

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Position Conflicts

The app automatically validates positions. If you see warnings:
- Change conflicting entity positions
- Entities cannot occupy the same cell
- Agent always starts at [0, 0]

### Memory Issues with Large Experiments

For grid size or gamma sensitivity analysis:
- Reduce `n_runs` parameter (default: 3)
- Lower `episodes` count
- Run one analysis at a time

### Empty or Single-Point Plots

If plots show only one point:
- Ensure episodes > 1
- Check console for error messages
- Verify training completed successfully

## ❓ FAQ

**Q: Can I change the colors in the visualization?**  
A: Yes! In the Streamlit interface, expand "GIF Color Customization" under Output Options.

**Q: How do I make the agent learn faster?**  
A: Increase alpha (learning rate to 0.2), reduce epsilon (exploration to 0.05), or use reward shaping.

**Q: Can I add more than one goal?**  
A: Yes! Set "Number of Goals" to any value 1-10 and configure each position.

**Q: What's the difference between static and dynamic entities?**  
A: Static entities stay in place; dynamic entities move randomly each step, making the environment non-stationary and harder to learn.

**Q: How do I interpret the training report?**  
A: Look for:
- Average reward trending upward = learning progress
- Success rate > 80% = excellent performance
- Final Q-delta < 1e-4 = converged
- Training time = benchmark for similar setups

**Q: Can I use this for research?**  
A: Yes! The code is provided for educational and research purposes. See [Citation](#citation).

**Q: Why does my agent keep hitting obstacles?**  
A: Try:
- Increase training episodes (2-3× current)
- Ensure goal is reachable (not blocked)
- Check max_steps is sufficient
- Try lower exploration rate (ε = 0.05)

**Q: What's the best gamma value?**  
A: Run gamma sensitivity analysis to find out! Generally:
- Simple environments: γ = 0.8-0.9
- Complex environments: γ = 0.9-0.99

## 💡 Tips for Best Results

1. **Start Small**: Begin with 5×5 grids to understand the system
2. **Use Defaults**: Default hyperparameters (α=0.1, γ=0.9, ε=0.1) work well
3. **Monitor Convergence**: Check Q-delta plot to see if training is complete
4. **Save Everything**: Enable all output options for complete documentation
5. **Experiment Systematically**: Change one parameter at a time
6. **Run Advanced Analysis**: Use grid size and gamma sensitivity for insights
7. **Read the Interpretations**: Each plot has detailed guidance in the app

## 📚 Documentation

- **README.md** (this file): Project overview and quick reference
- **USER_GUIDE.md**: Complete 10-section guide with detailed explanations
- **GITHUB_SETUP.md**: Step-by-step instructions for GitHub setup
- **CONTRIBUTING.md**: Guidelines for contributing to the project

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for contribution:
- New environment features
- Additional RL algorithms
- Visualization improvements
- Performance optimizations
- Documentation enhancements
- Bug fixes

## 📖 Citation

If using this project for academic purposes, please reference:

```bibtex
@software{gridworld_qlearning_2024,
  author = {Your Name},
  title = {GridWorld Q-Learning Training Platform},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/gridworld-qlearning}
}
```

**Q-Learning Algorithm**: Watkins, C.J.C.H., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/) (OpenAI Gym successor)
- UI powered by [Streamlit](https://streamlit.io/)
- Visualization using [Matplotlib](https://matplotlib.org/) and [Pygame](https://www.pygame.org/)

## 📧 Contact

For questions, suggestions, or issues:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/gridworld-qlearning/issues)
- **Email**: your.email@example.com

---

**🌟 If you find this project useful, please consider giving it a star!**

=======
>>>>>>> c508dcdad35659ecf690dcd614a41bb5b702b04f
**Made with ❤️ for the Reinforcement Learning community**