# 📚 GridWorld Q-Learning - Complete User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Environment Configuration](#environment-configuration)
4. [Training Configuration](#training-configuration)
5. [Understanding Q-Learning](#understanding-q-learning)
6. [Interpreting Results](#interpreting-results)
7. [Advanced Analysis](#advanced-analysis)
8. [Best Practices](#best-practices)
9. [FAQ](#faq)

---

## 1. Introduction

### What is GridWorld Q-Learning?

This platform allows you to train intelligent agents to navigate grid-based environments using **Q-Learning**, a foundational reinforcement learning algorithm. The agent learns through trial and error, discovering optimal paths from start to goal while avoiding obstacles.

### Key Concepts

**GridWorld**: A 2D grid where:
- **Agent** (green circle): Learns to navigate
- **Goal** (red square): Target destination
- **Obstacles** (black squares): Must be avoided
- **Other Agents** (dark green): Competing agents that cause collisions

**Q-Learning**: The agent maintains a "Q-table" that stores expected rewards for each action in each state, learning the optimal policy through experience.

---

## 2. Getting Started

### Launching the Application

```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501`

### Interface Layout

- **Left Sidebar**: All configuration options
- **Main Area**: Configuration summary and results
- **Bottom**: Training visualizations and reports after training

---

## 3. Environment Configuration

### 3.1 Grid Dimensions

**Grid Rows & Columns** (2-20 each)

- **Small grids (3×3 to 5×5)**: Fast training, simple problems
- **Medium grids (6×6 to 10×10)**: Balanced complexity
- **Large grids (11×11+)**: Slow training, complex navigation

**💡 Recommendation**: Start with 5×5 for learning, scale up as needed.

### 3.2 Goals

**Number of Goals** (1-10)

Multiple goals allow the agent to reach any one to succeed.

**Goal Placement Options**:

1. **Default (bottom-right)**: ✅ Recommended for beginners
   - Automatically places goal at grid position [rows-1, cols-1]
   - Classic setup for navigation problems

2. **Custom positions**: 
   - Manually specify row and column for each goal
   - Useful for complex scenarios

**Goals Dynamic**: ☑️ Check to make goals move randomly each step
- **Static goals**: Easier learning, faster convergence
- **Dynamic goals**: Harder problem, agent must track moving target

**⚠️ Warning**: Position conflicts will show orange warnings. Fix before training.

### 3.3 Obstacles

**Number of Obstacles** (0-20)

Obstacles block movement and give penalties when bumped.

**Obstacle Placement**:
- Specify row and column for each obstacle
- Avoid blocking the only path to goal
- Consider symmetric patterns for balanced difficulty

**Obstacles Dynamic**: ☑️ Check for moving obstacles
- Creates non-stationary environment
- Significantly increases difficulty
- Agent must learn adaptive strategies

**💡 Strategy**:
- Start with 2-3 static obstacles
- Place obstacles to create interesting but solvable mazes
- Avoid clustering all obstacles in one area

### 3.4 Other Agents

**Number of Other Agents** (0-10)

Competing agents that cause the episode to terminate on collision (large penalty: -1.0).

**Use Cases**:
- Multi-agent scenarios
- Collision avoidance training
- Competitive environments

**Other Agents Dynamic**: ☑️ Check for moving agents
- Agents move randomly but avoid collisions with obstacles/goals
- Creates unpredictable environment

**⚠️ Important**: Colliding with other agents terminates the episode with -1 reward.

### 3.5 Validation Rules

✅ **Valid Configurations**:
- No two entities on same cell
- Agent starts at [0, 0] (top-left)
- At least one path exists to goal

❌ **Invalid Configurations**:
- Overlapping entities
- Goal at agent start position
- Completely blocked goals

---

## 4. Training Configuration

### 4.1 Number of Episodes

**Episodes** (10-10,000)

One episode = agent attempts navigation from start until:
- Goal reached (success)
- Collision with other agent (failure)
- Max steps exceeded (timeout)

**Guidelines**:
- **Small grids (3×3 - 5×5)**: 100-300 episodes
- **Medium grids (6×6 - 8×8)**: 300-500 episodes
- **Large grids (9×9+)**: 500-1,000+ episodes
- **Dynamic environments**: Add 50-100% more episodes

### 4.2 Hyperparameters

#### Learning Rate (Alpha, α)

**Range**: 0.0 - 1.0 | **Default**: 0.1

Controls how much new information overrides old knowledge.

- **α = 0.0**: Agent never learns (not useful)
- **α = 0.1**: Slow, stable learning ✅ Recommended
- **α = 0.5**: Moderate learning speed
- **α = 1.0**: Fast but unstable learning

**When to adjust**:
- ↑ α (0.2-0.3): If learning is too slow
- ↓ α (0.05): If learning is unstable/oscillating

#### Discount Factor (Gamma, γ)

**Range**: 0.0 - 1.0 | **Default**: 0.9

Controls importance of future rewards vs immediate rewards.

- **γ = 0.0**: Only immediate rewards matter (myopic)
- **γ = 0.5**: Balanced short/long term
- **γ = 0.9**: Strong future planning ✅ Recommended
- **γ = 0.99**: Maximum long-term optimization

**When to adjust**:
- ↑ γ (0.95-0.99): For optimal long-term paths
- ↓ γ (0.7-0.8): For faster convergence on simpler problems

**💡 Trade-off**: Higher γ = better policies but slower learning

#### Exploration Rate (Epsilon, ε)

**Range**: 0.0 - 1.0 | **Default**: 0.1

Probability of random action (exploration) vs best known action (exploitation).

- **ε = 0.0**: Pure exploitation (may miss optimal paths)
- **ε = 0.1**: Balanced exploration ✅ Recommended
- **ε = 0.5**: High exploration
- **ε = 1.0**: Random actions only (no learning)

**Note**: The implementation uses epsilon decay, automatically reducing ε during training.

### 4.3 Max Steps Per Episode

**Default**: rows × cols × 4

Maximum actions before episode timeout.

- **Too low**: Agent can't reach goal (premature timeout)
- **Too high**: Wastes computation on failed episodes
- **Default formula**: Usually appropriate

**When to adjust**:
- ↑ max steps: If agent frequently times out despite making progress
- ↓ max steps: To force more efficient path learning

### 4.4 Reward Shaping

☑️ **Enable reward shaping**

Adds small penalty (-0.01) for each step to encourage efficiency.

**Effects**:
- **Enabled**: Agent learns shorter paths (more negative for longer episodes)
- **Disabled**: Agent only cares about reaching goal (may take longer paths)

**💡 Recommendation**: Enable for environments where path length matters.

---

## 5. Understanding Q-Learning

### How It Works

1. **Initialize**: Q-table starts with zeros
2. **Explore**: Agent tries actions (epsilon-greedy)
3. **Experience**: Agent observes result (reward, next state)
4. **Learn**: Update Q-value using Bellman equation:
   ```
   Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
   ```
5. **Repeat**: For many episodes until convergence

### Reward Structure

| Event | Reward | Outcome |
|-------|--------|---------|
| Reach goal | +1.0 | Episode ends (success) |
| Collide with other agent | -1.0 | Episode ends (failure) |
| Bump into obstacle | -0.1 | Episode continues |
| Bump into wall | -0.1 | Episode continues |
| Normal move | 0.0 | Episode continues |
| Step (if reward shaping) | -0.01 | Additional penalty |

### Actions

The agent can take 4 actions:
- **0**: Move Up (↑)
- **1**: Move Down (↓)
- **2**: Move Left (←)
- **3**: Move Right (→)

Invalid moves (into walls) keep agent in place with penalty.

---

## 6. Interpreting Results

### 6.1 Training Metrics

**Episodes**: Total training episodes
- Should equal your configured amount

**Avg Reward**: Mean reward across all episodes
- **> 0.5**: Excellent performance
- **0.0 - 0.5**: Good performance
- **< 0.0**: Poor performance, needs more training

**Goal Reached**: Successful episodes
- Compare to total episodes for success rate

**Success Rate**: Percentage reaching goal
- **> 80%**: Excellent
- **50-80%**: Good
- **< 50%**: Needs improvement

**Training Time**: Total seconds
- Use to estimate larger experiments

**Best Reward**: Highest single episode reward
- Usually close to +1.0 for successful runs

**Final Q-Delta**: Q-table stability at end
- **< 10⁻⁴**: Well converged ✅
- **> 10⁻³**: May need more episodes

### 6.2 Episode Rewards Plot

**What to Look For**:

✅ **Good Learning Patterns**:
- Upward trend over time
- Values stabilizing near +1.0
- Decreasing variance

❌ **Problem Patterns**:
- Flat negative line → Not learning (check hyperparameters)
- High oscillation → Too much exploration or unstable α
- Sudden drops → Environment may be too stochastic

**Example Interpretation**:
```
Episodes 1-50:   Rewards around -0.5 (random exploration)
Episodes 51-100: Rewards improving to 0.0 (finding paths)
Episodes 101+:   Rewards near +0.8 (optimized paths with occasional exploration)
```

### 6.3 Q-Delta Convergence Plot

**What It Shows**: Maximum change in Q-table between episodes (log scale)

✅ **Good Convergence**:
- Smooth decreasing trend
- Reaches < 10⁻⁴ 
- Stays flat at low values

❌ **Poor Convergence**:
- Oscillating wildly
- Not decreasing
- Large final delta

**Phases**:
1. **High delta (10⁰ - 10⁻¹)**: Initial learning, exploring state space
2. **Decreasing (10⁻² - 10⁻³)**: Refining policies
3. **