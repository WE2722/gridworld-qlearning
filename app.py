"""
GridWorld Q-Learning Streamlit Application - COMPLETE VERSION
Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Page configuration
st.set_page_config(
    page_title="GridWorld Q-Learning",
    page_icon="🤖",
    layout="wide"
)

# Import with error handling
try:
    from Livrable_3_2_WIAME_EL_HAFID import (
        GridWorldEnv,
        QLearningAgent,
        export_training_gif_custom_colors,
        convergence_vs_grid,
        visualize_gamma_sensitivity,
    )
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Make sure 'Livrable_3_2_WIAME_EL_HAFID.py' exists in your repository")
    st.stop()

def create_folders():
    """Create necessary output folders"""
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/gifs', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

def generate_report(config, rewards, deltas, training_time, goal_reached_count):
    """Generate training report - COMPLETE VERSION"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'output/reports/report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GridWorld Q-Learning Training Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ENVIRONMENT CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Grid Size: {config['rows']}x{config['cols']}\n")
        f.write(f"Total Cells: {config['rows'] * config['cols']}\n")
        f.write(f"Goals: {config['n_goals']} at {config['goals_pos']}\n")
        f.write(f"  Dynamic: {config['goals_dynamic']}\n")
        f.write(f"Obstacles: {config['n_obstacles']}")
        if config['obstacles_pos']:
            f.write(f" at {config['obstacles_pos']}\n")
        else:
            f.write("\n")
        f.write(f"  Dynamic: {config['obstacles_dynamic']}\n")
        f.write(f"Other Agents: {config['n_other_agents']}")
        if config['other_agents_pos']:
            f.write(f" at {config['other_agents_pos']}\n")
        else:
            f.write("\n")
        f.write(f"  Dynamic: {config['other_agents_dynamic']}\n")
        f.write(f"Max Steps per Episode: {config['max_steps']}\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Episodes: {config['episodes']}\n")
        f.write(f"Learning Rate (alpha): {config['alpha']}\n")
        f.write(f"Discount Factor (gamma): {config['gamma']}\n")
        f.write(f"Exploration Rate (epsilon): {config['epsilon']}\n")
        f.write(f"Reward Shaping: {config['reward_shaping']}\n\n")
        
        f.write("REWARD STRUCTURE\n")
        f.write("-" * 70 + "\n")
        f.write("  Reach Goal: +1.0 (episode terminates)\n")
        f.write("  Collide with Other Agent: -1.0 (episode terminates)\n")
        f.write("  Bump into Obstacle: -0.1 (episode continues)\n")
        f.write("  Bump into Wall: -0.1 (episode continues)\n")
        f.write("  Normal Move: 0.0 (episode continues)\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Episodes Completed: {len(rewards)}\n")
        f.write(f"Goal Reached: {goal_reached_count} times ({goal_reached_count/len(rewards)*100:.1f}%)\n\n")
        
        if rewards:
            f.write("Episode Rewards:\n")
            f.write(f"  Average: {np.mean(rewards):.4f}\n")
            f.write(f"  Final Episode: {rewards[-1]:.4f}\n")
            f.write(f"  Best Episode: {max(rewards):.4f} (Episode {rewards.index(max(rewards))+1})\n")
            f.write(f"  Worst Episode: {min(rewards):.4f} (Episode {rewards.index(min(rewards))+1})\n")
            f.write(f"  Standard Deviation: {np.std(rewards):.4f}\n\n")
            
            if len(rewards) >= 10:
                first_10_avg = np.mean(rewards[:10])
                last_10_avg = np.mean(rewards[-10:])
                improvement = last_10_avg - first_10_avg
                f.write(f"Learning Progress:\n")
                f.write(f"  First 10 Episodes Avg: {first_10_avg:.4f}\n")
                f.write(f"  Last 10 Episodes Avg: {last_10_avg:.4f}\n")
                f.write(f"  Improvement: {improvement:+.4f} ({improvement/abs(first_10_avg)*100:+.1f}%)\n\n")
        
        if deltas:
            f.write("Q-Table Convergence:\n")
            f.write(f"  Initial Q-Delta: {deltas[0]:.6e}\n")
            f.write(f"  Final Q-Delta: {deltas[-1]:.6e}\n")
            f.write(f"  Converged (delta < 1e-4): {'Yes' if deltas[-1] < 1e-4 else 'No'}\n")
            if deltas[-1] < 1e-4:
                for i, d in enumerate(deltas):
                    if d < 1e-4:
                        f.write(f"  Convergence Episode: {i+1}\n")
                        break
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n\n")
        f.write("Episode Rewards:\n")
        f.write("  - Positive trend = agent is learning\n")
        f.write("  - Values approaching +1.0 = reaching goal efficiently\n")
        f.write("  - Negative values = hitting obstacles or failing to reach goal\n\n")
        f.write("Goal Reached Percentage:\n")
        f.write("  - >80% = excellent performance\n")
        f.write("  - 50-80% = good performance\n")
        f.write("  - <50% = needs more training or parameter tuning\n\n")
        f.write("Q-Delta Convergence:\n")
        f.write("  - Decreasing trend = Q-table stabilizing\n")
        f.write("  - Delta < 1e-4 = agent has learned optimal policy\n")
        f.write("  - High final delta = may need more episodes\n\n")
        
        f.write("=" * 70 + "\n")
    
    return report_path

def main():
    st.title("🤖 GridWorld Q-Learning Training")
    st.markdown("Configure and train a Q-Learning agent in a customizable grid environment")
    
    create_folders()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Quick presets
        st.subheader("🎯 Quick Start")
        preset = st.selectbox("Choose preset:", [
            "Custom",
            "🟢 Easy (3×3, 200 episodes)",
            "🟡 Medium (5×5, 300 episodes)",
            "🔴 Hard (7×7, 500 episodes)"
        ])
        
        if preset == "🟢 Easy (3×3, 200 episodes)":
            rows, cols, episodes = 3, 3, 200
            n_goals, n_obstacles = 1, 1
            use_default_goal = True
        elif preset == "🟡 Medium (5×5, 300 episodes)":
            rows, cols, episodes = 5, 5, 300
            n_goals, n_obstacles = 1, 2
            use_default_goal = True
        elif preset == "🔴 Hard (7×7, 500 episodes)":
            rows, cols, episodes = 7, 7, 500
            n_goals, n_obstacles = 1, 3
            use_default_goal = True
        else:
            # Environment
            st.subheader("🌍 Environment")
            rows = st.number_input("Grid Rows", min_value=2, max_value=20, value=5)
            cols = st.number_input("Grid Columns", min_value=2, max_value=20, value=5)
            
            # Goals
            st.markdown("**Goals**")
            n_goals = st.number_input("Number of Goals", min_value=1, max_value=10, value=1)
            use_default_goal = st.checkbox("Place goal at bottom-right", value=True)
            
            # Obstacles
            st.markdown("**Obstacles**")
            n_obstacles = st.number_input("Number of Obstacles", min_value=0, max_value=20, value=2)
            
            # Training
            st.subheader("🎓 Training")
            episodes = st.number_input("Training Episodes", min_value=10, max_value=10000, value=200)
        
        # Configure goals
        if use_default_goal:
            goals_pos = [[rows-1, cols-1]]
        else:
            goals_pos = []
            used_positions = set()
            used_positions.add((0, 0))
            
            for i in range(n_goals):
                col1, col2 = st.columns(2)
                with col1:
                    g_row = st.number_input(f"Goal {i+1} Row", 0, rows-1, min(i, rows-1), key=f"g_row_{i}")
                with col2:
                    g_col = st.number_input(f"Goal {i+1} Col", 0, cols-1, min(i, cols-1), key=f"g_col_{i}")
                
                pos = (g_row, g_col)
                if pos in used_positions:
                    st.warning(f"⚠️ Goal {i+1} at {pos} conflicts with another entity!")
                else:
                    used_positions.add(pos)
                
                goals_pos.append([g_row, g_col])
        
        goals_dynamic = st.checkbox("Goals move randomly", value=False)
        
        # Configure obstacles
        obstacles_pos = []
        if n_obstacles > 0:
            if not use_default_goal:
                used_positions_obs = used_positions.copy()
            else:
                used_positions_obs = {(0, 0), (rows-1, cols-1)}
            
            for i in range(n_obstacles):
                col1, col2 = st.columns(2)
                with col1:
                    o_row = st.number_input(f"Obstacle {i+1} Row", 0, rows-1, 
                                          min(1+i, rows-1), key=f"o_row_{i}")
                with col2:
                    o_col = st.number_input(f"Obstacle {i+1} Col", 0, cols-1, 
                                          min(2+i, cols-1), key=f"o_col_{i}")
                
                pos = (o_row, o_col)
                if pos in used_positions_obs:
                    st.warning(f"⚠️ Obstacle {i+1} at {pos} conflicts with another entity!")
                else:
                    used_positions_obs.add(pos)
                
                obstacles_pos.append([o_row, o_col])
            obstacles_dynamic = st.checkbox("Obstacles move randomly", value=False)
        else:
            obstacles_pos = None
            obstacles_dynamic = False
            used_positions_obs = used_positions.copy() if not use_default_goal else {(0, 0), (rows-1, cols-1)}
        
        # Other Agents
        st.markdown("**Other Agents**")
        n_other_agents = st.number_input("Number of Other Agents", min_value=0, max_value=10, value=0)
        other_agents_pos = []
        other_agents_dynamic = False
        if n_other_agents > 0:
            used_positions_agents = used_positions_obs.copy()
            
            for i in range(n_other_agents):
                col1, col2 = st.columns(2)
                with col1:
                    a_row = st.number_input(f"Agent {i+1} Row", 0, rows-1, 
                                          min(2+i, rows-1), key=f"a_row_{i}")
                with col2:
                    a_col = st.number_input(f"Agent {i+1} Col", 0, cols-1, 
                                          min(1+i, cols-1), key=f"a_col_{i}")
                
                pos = (a_row, a_col)
                if pos in used_positions_agents:
                    st.warning(f"⚠️ Agent {i+1} at {pos} conflicts with another entity!")
                else:
                    used_positions_agents.add(pos)
                
                other_agents_pos.append([a_row, a_col])
            other_agents_dynamic = st.checkbox("Other agents move randomly", value=False)
        else:
            other_agents_pos = None
        
        # Training Configuration
        st.subheader("🎓 Training")
        
        use_default_params = st.checkbox("Use default hyperparameters", value=True)
        if use_default_params:
            alpha = 0.1
            gamma = 0.9
            epsilon = 0.1
        else:
            alpha = st.slider("Learning Rate (alpha)", 0.0, 1.0, 0.1, 0.01)
            gamma = st.slider("Discount Factor (gamma)", 0.0, 1.0, 0.9, 0.01)
            epsilon = st.slider("Exploration Rate (epsilon)", 0.0, 1.0, 0.1, 0.01)
        
        max_steps_default = max(100, rows * cols * 4)
        use_default_steps = st.checkbox(f"Use default max steps ({max_steps_default})", value=True)
        max_steps = None if use_default_steps else st.number_input("Max Steps per Episode", 
                                                                   min_value=10, value=max_steps_default)
        
        reward_shaping = st.checkbox("Enable reward shaping", value=False)
        
        # Output Options
        st.subheader("💾 Output Options")
        save_model = st.checkbox("Save trained model", value=True)
        save_gif = st.checkbox("Export training GIF", value=True)
        if save_gif:
            gif_episodes = st.number_input("GIF Episodes", 1, episodes, min(50, episodes))
            gif_fps = st.number_input("GIF FPS", 1, 30, 5)
            
            st.markdown("**GIF Color Customization**")
            col1, col2 = st.columns(2)
            with col1:
                agent_color = st.color_picker("Main Agent Color", "#00FF00")
                goal_color = st.color_picker("Goal Color", "#FF0000")
            with col2:
                obstacle_color = st.color_picker("Obstacle Color", "#000000")
                other_agent_color = st.color_picker("Other Agents Color", "#009600")
        
        generate_report_opt = st.checkbox("Generate training report", value=True)
        
        # Plot Options
        st.subheader("📊 Plots")
        plot_rewards = st.checkbox("Episode Rewards", value=True)
        plot_qdeltas = st.checkbox("Q-Delta Convergence", value=True)
        plot_combined = st.checkbox("Combined Plot", value=True)
        
        st.markdown("**Advanced Analysis**")
        run_grid_analysis = st.checkbox("Grid Size Analysis", value=False)
        run_gamma_analysis = st.checkbox("Gamma Sensitivity", value=False)
        
        if run_gamma_analysis:
            save_gamma_models = st.checkbox("Save models for each gamma", value=True)
        else:
            save_gamma_models = False
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Configuration Summary")
        st.markdown(f"""
        **Environment:** {rows}×{cols} grid  
        **Goals:** {n_goals} {'(dynamic)' if goals_dynamic else '(static)'}  
        **Obstacles:** {n_obstacles} {'(dynamic)' if obstacles_dynamic else '(static)'}  
        **Other Agents:** {n_other_agents} {'(dynamic)' if other_agents_dynamic else '(static)'}  
        **Episodes:** {episodes}  
        **Hyperparameters:** α={alpha}, γ={gamma}, ε={epsilon}
        """)
    
    with col2:
        st.subheader("🎯 Output Files")
        st.markdown("""
        - `output/models/` - Trained models
        - `output/plots/` - Generated plots
        - `output/gifs/` - Training animations
        - `output/reports/` - Training reports
        """)
    
    # Train button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        
        config = {
            'rows': rows, 'cols': cols,
            'n_goals': n_goals, 'goals_pos': goals_pos, 'goals_dynamic': goals_dynamic,
            'n_obstacles': n_obstacles, 'obstacles_pos': obstacles_pos, 'obstacles_dynamic': obstacles_dynamic,
            'n_other_agents': n_other_agents, 'other_agents_pos': other_agents_pos, 
            'other_agents_dynamic': other_agents_dynamic,
            'max_steps': max_steps if max_steps else max_steps_default,
            'episodes': episodes, 'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon,
            'reward_shaping': reward_shaping
        }
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Validation
            status_text.text("Validating configuration...")
            all_positions = set()
            agent_start = (0, 0)
            all_positions.add(agent_start)
            
            for g in goals_pos:
                pos = (g[0], g[1])
                if pos in all_positions:
                    st.error(f"Position conflict at {pos}!")
                    st.stop()
                all_positions.add(pos)
            
            if obstacles_pos:
                for o in obstacles_pos:
                    pos = (o[0], o[1])
                    if pos in all_positions:
                        st.error(f"Position conflict at {pos}!")
                        st.stop()
                    all_positions.add(pos)
            
            if other_agents_pos:
                for a in other_agents_pos:
                    pos = (a[0], a[1])
                    if pos in all_positions:
                        st.error(f"Position conflict at {pos}!")
                        st.stop()
                    all_positions.add(pos)
            
            # Create environment
            status_text.text("Creating environment...")
            env = GridWorldEnv(
                rows=rows, cols=cols,
                n_goals=n_goals, goals_pos=goals_pos, goals_dynamic=goals_dynamic,
                n_obstacles=n_obstacles, obstacles_pos=obstacles_pos, obstacles_dynamic=obstacles_dynamic,
                n_other_agents=n_other_agents, other_agents_pos=other_agents_pos,
                other_agents_dynamic=other_agents_dynamic,
                render=False, max_steps=max_steps, seed=123
            )
            
            agent = QLearningAgent(env, episodes=episodes, alpha=alpha, gamma=gamma, 
                                 epsilon=epsilon, reward_shaping=reward_shaping)
            
            # Training
            status_text.text("Training agent...")
            progress_bar.progress(0.2)
            
            import time
            start_time = time.time()
            
            goal_reached_count = 0
            rewards_list = []
            deltas_list = []
            
            for ep in range(episodes):
                state, _ = env.reset(seed=123+ep)
                done = False
                t = 0
                ep_reward = 0
                Q_prev = agent.Q.copy()
                
                while not done and t < (max_steps if max_steps else max_steps_default):
                    agent.epsilon = max(0.05, 0.2*100/(100+t))
                    if np.random.rand() < agent.epsilon:
                        action = env.action_space.sample()
                    else:
                        action = int(np.argmax(agent.Q[state[0], state[1]]))
                        
                    next_state, reward, done, _, _ = env.step(action)
                    
                    if agent.reward_shaping:
                        reward -= 0.01
                        
                    best_next = np.max(agent.Q[next_state[0], next_state[1]])
                    agent.Q[state[0], state[1], action] += agent.alpha * (
                        reward + agent.gamma * best_next - agent.Q[state[0], state[1], action]
                    )
                    
                    state = next_state.copy()
                    ep_reward += reward
                    t += 1
                    
                    if reward == 1.0:
                        goal_reached_count += 1
                        break
                
                rewards_list.append(ep_reward)
                delta = np.max(np.abs(agent.Q - Q_prev))
                deltas_list.append(delta)
                agent.policy = np.argmax(agent.Q, axis=2)
            
            rewards = rewards_list
            deltas = deltas_list
            training_time = time.time() - start_time
            
            progress_bar.progress(0.5)
            status_text.text("Training complete!")
            
            # Save model
            if save_model:
                status_text.text("Saving model...")
                model_path = f'output/models/model_{rows}x{cols}_ep{episodes}'
                agent.save_model(model_path)
            
            progress_bar.progress(0.6)
            
            # Export GIF
            if save_gif:
                status_text.text(f"Exporting GIF ({gif_episodes} episodes)...")
                gif_env = GridWorldEnv(
                    rows=rows, cols=cols, n_goals=n_goals, goals_pos=goals_pos, goals_dynamic=goals_dynamic,
                    n_obstacles=n_obstacles, obstacles_pos=obstacles_pos, obstacles_dynamic=obstacles_dynamic,
                    n_other_agents=n_other_agents, other_agents_pos=other_agents_pos,
                    other_agents_dynamic=other_agents_dynamic,
                    render=False, max_steps=max_steps, seed=123
                )
                gif_agent = QLearningAgent(gif_env, episodes=gif_episodes, alpha=alpha, 
                                          gamma=gamma, epsilon=epsilon, reward_shaping=reward_shaping)
                gif_path = f'output/gifs/training_{rows}x{cols}_ep{gif_episodes}.gif'
                
                custom_colors = {
                    'agent': agent_color,
                    'goal': goal_color,
                    'obstacle': obstacle_color,
                    'other_agent': other_agent_color
                }
                
                try:
                    export_training_gif_custom_colors(gif_env, gif_agent, episodes=gif_episodes, 
                                                     out_path=gif_path, fps=gif_fps, colors=custom_colors)
                except Exception as e:
                    st.warning(f"GIF export failed: {e}")
                gif_env.close()
            
            progress_bar.progress(0.7)
            
            # Generate plots
            status_text.text("Generating plots...")
            
            if plot_rewards and len(rewards) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, len(rewards)+1), rewards, '-', color='tab:blue', linewidth=2)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Episode Reward')
                ax.set_title('Episode Rewards Over Training')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'output/plots/rewards_{rows}x{cols}.png', dpi=150)
                plt.close()
            
            if plot_qdeltas and len(deltas) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                safe_deltas = [max(d, 1e-12) for d in deltas]
                ax.plot(range(1, len(safe_deltas)+1), safe_deltas, '-', color='tab:orange', linewidth=2)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Max Q-table Delta')
                ax.set_yscale('log')
                ax.set_title('Q-table Convergence (Max Delta per Episode)')
                ax.grid(True, which='both', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'output/plots/qdeltas_{rows}x{cols}.png', dpi=150)
                plt.close()
            
            if plot_combined and len(rewards) > 0 and len(deltas) > 0:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(range(1, len(rewards)+1), rewards, '-', color='tab:blue', 
                        linewidth=2, label='Episode Reward')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Episode Reward', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)
                
                ax2 = ax1.twinx()
                safe_deltas = [max(d, 1e-12) for d in deltas]
                ax2.plot(range(1, len(safe_deltas)+1), safe_deltas, '-', color='tab:orange', 
                        linewidth=2, label='Max Q Delta')
                ax2.set_ylabel('Max Q Delta (log scale)', color='tab:orange')
                ax2.set_yscale('log')
                ax2.tick_params(axis='y', labelcolor='tab:orange')
                ax2.legend(loc='upper right')
                
                plt.title('Training Convergence: Rewards and Q-table Stability')
                plt.tight_layout()
                plt.savefig(f'output/plots/combined_{rows}x{cols}.png', dpi=150)
                plt.close()
            
            progress_bar.progress(0.8)
            
            # Grid analysis
            if run_grid_analysis:
                status_text.text("Running grid size analysis...")
                grid_sizes = [3, 5, 7, 9] if rows <= 9 else [3, 5, 7]
                convergence_vs_grid(grid_sizes, base_goal_rel=(1.0, 1.0),
                                  obstacle_rel_positions=[(0.25, 0.4), (0.4, 0.6)] if n_obstacles >= 2 else None,
                                  episodes=min(100, episodes), n_runs=3, conv_tol=1e-4)
                if os.path.exists('convergence_vs_grid.png'):
                    import shutil
                    shutil.move('convergence_vs_grid.png', 'output/plots/convergence_vs_grid.png')
            
            # Gamma analysis
            if run_gamma_analysis:
                status_text.text("Running gamma sensitivity analysis...")
                
                def env_factory():
                    return GridWorldEnv(
                        rows=min(rows, 5), cols=min(cols, 5),
                        n_goals=1, goals_pos=[[min(rows-1, 4), min(cols-1, 4)]], goals_dynamic=False,
                        n_obstacles=min(n_obstacles, 2), 
                        obstacles_pos=[[1,2],[2,3]] if n_obstacles >= 2 else None,
                        obstacles_dynamic=False, n_other_agents=0, render=False, seed=None
                    )
                
                gammas = [0.0, 0.3, 0.6, 0.9, 0.99]
                
                if save_gamma_models:
                    for g in gammas:
                        g_env = env_factory()
                        g_agent = QLearningAgent(g_env, episodes=min(150, episodes), gamma=g)
                        g_agent.train()
                        g_agent.save_model(f'output/models/model_gamma_{g:.2f}')
                        g_env.close()
                
                visualize_gamma_sensitivity(env_factory, gammas, episodes=min(150, episodes))
                
                if os.path.exists('gamma_sensitivity_conv_vs_reward.png'):
                    import shutil
                    shutil.move('gamma_sensitivity_conv_vs_reward.png', 
                               'output/plots/gamma_sensitivity_conv_vs_reward.png')
                if os.path.exists('gamma_sensitivity_conv.png'):
                    import shutil
                    shutil.move('gamma_sensitivity_conv.png', 'output/plots/gamma_sensitivity_conv.png')
            
            progress_bar.progress(0.9)
            
            # Generate report
            if generate_report_opt:
                status_text.text("Generating training report...")
                report_path = generate_report(config, rewards, deltas, training_time, goal_reached_count)
            
            progress_bar.progress(1.0)
            status_text.text("✅ All tasks completed!")
            
            env.close()
            
            # Display results
            st.success("Training completed successfully!")
            
            st.subheader("📊 Training Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Episodes", len(rewards))
            with col2:
                st.metric("Avg Reward", f"{np.mean(rewards):.3f}" if rewards else "N/A")
            with col3:
                st.metric("Goal Reached", f"{goal_reached_count}/{len(rewards)}")
            with col4:
                st.metric("Success Rate", f"{goal_reached_count/len(rewards)*100:.1f}%" if rewards else "N/A")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Time", f"{training_time:.1f}s")
            with col2:
                st.metric("Best Reward", f"{max(rewards):.3f}" if rewards else "N/A")
            with col3:
                st.metric("Final Q-Delta", f"{deltas[-1]:.2e}" if deltas else "N/A")
            
            # Display plots with explanations
            st.subheader("📈 Training Visualizations")
            
            if plot_rewards and os.path.exists(f'output/plots/rewards_{rows}x{cols}.png'):
                st.markdown("### Episode Rewards")
                st.markdown("""
                **What this shows:** Cumulative reward for each training episode.
                
                **How to interpret:**
                - **Upward trend** → Agent is learning and improving
                - **Values near +1.0** → Agent reaches goal efficiently with minimal bumps
                - **Negative values** → Agent hits many obstacles or doesn't reach goal
                - **Stabilization** → Agent has learned a consistent strategy
                """)
                st.image(f'output/plots/rewards_{rows}x{cols}.png')
            
            if plot_qdeltas and os.path.exists(f'output/plots/qdeltas_{rows}x{cols}.png'):
                st.markdown("### Q-Table Convergence")
                st.markdown("""
                **What this shows:** Maximum change in Q-values between episodes (log scale).
                
                **How to interpret:**
                - **Decreasing trend** → Q-table is stabilizing, agent is converging
                - **Delta < 10⁻⁴** → Agent has learned near-optimal policy
                - **Flat line** → No more learning occurring
                - **Oscillations** → Agent still exploring or environment is too complex
                """)
                st.image(f'output/plots/qdeltas_{rows}x{cols}.png')
            
            if plot_combined and os.path.exists(f'output/plots/combined_{rows}x{cols}.png'):
                st.markdown("### Combined Analysis")
                st.markdown("""
                **What this shows:** Rewards and Q-convergence on the same timeline.
                
                **How to interpret:**
                - **Both improving together** → Healthy learning progress
                - **Rewards plateau but Q still changing** → Fine-tuning strategy
                - **Rewards unstable but Q converged** → Stochastic environment or exploration
                """)
                st.image(f'output/plots/combined_{rows}x{cols}.png')
            
            # Advanced Analysis Results
            if run_grid_analysis and os.path.exists('output/plots/convergence_vs_grid.png'):
                st.markdown("---")
                st.markdown("### 📐 Grid Size Analysis")
                st.markdown("""
                **What this shows:** How grid dimensions affect learning convergence speed.
                
                **How to interpret:**
                - **Y-axis:** Number of episodes needed for Q-table to converge (smaller is better)
                - **X-axis:** Grid dimensions (e.g., 3×3, 5×5, 7×7, 9×9)
                - **Error bars:** Show variance across multiple runs (smaller bars = more consistent)
                
                **Expected pattern:**
                - **Increasing trend** → Larger grids need more episodes (larger state space)
                - **Exponential growth** → Complexity grows quickly with grid size
                - **Large error bars** → Environment has high randomness or needs more runs
                
                **What this tells you:**
                - **Scalability:** How well Q-learning scales to larger environments
                - **Complexity cost:** Computational burden of increasing grid size
                - **Training requirements:** Episodes needed for different problem sizes
                """)
                st.image('output/plots/convergence_vs_grid.png')
                
                with st.expander("💡 Practical Implications"):
                    st.markdown("""
                    - If convergence episodes grow slowly → Q-learning handles this problem well
                    - If convergence episodes explode → Consider function approximation (Deep Q-Learning)
                    - Use this to estimate training time for production environments
                    - Compare multiple algorithms using this metric
                    """)
            
            if run_gamma_analysis:
                if os.path.exists('output/plots/gamma_sensitivity_conv_vs_reward.png'):
                    st.markdown("---")
                    st.markdown("### 🎯 Gamma Sensitivity Analysis")
                    st.markdown("""
                    **What this shows:** How the discount factor (γ) affects learning performance.
                    
                    **Understanding Gamma (γ):**
                    - **γ = 0.0:** Agent only cares about immediate rewards (myopic)
                    - **γ = 0.5:** Moderate balance between short and long-term
                    - **γ = 0.9:** Strong preference for long-term rewards (common default)
                    - **γ = 0.99:** Very long-term planning (near-optimal paths)
                    
                    **Two metrics plotted:**
                    1. **Convergence Episode (left/blue):** Episodes to converge
                    2. **Average Reward (right/orange):** Quality of learned policy
                    
                    **How to interpret:**
                    
                    **Convergence Speed:**
                    - **Low γ (0.0-0.3)** → Fast convergence (simple policies)
                    - **High γ (0.9-0.99)** → Slower convergence (complex planning)
                    
                    **Reward Quality:**
                    - **Low γ** → Lower rewards (short-sighted decisions)
                    - **Optimal γ** → Highest rewards (best balance)
                    - **Too high γ** → May overfit or struggle to converge
                    
                    **Sweet spot:** Where both curves are favorable
                    """)
                    st.image('output/plots/gamma_sensitivity_conv_vs_reward.png')
                    
                    with st.expander("💡 Choosing the Right Gamma"):
                        st.markdown("""
                        **For your environment:**
                        - **Short episodes/simple goals:** Try γ = 0.7 - 0.9
                        - **Long episodes/complex paths:** Try γ = 0.9 - 0.99
                        - **Real-time systems:** Lower γ for faster convergence
                        - **Optimal performance:** Higher γ but longer training
                        
                        **Trade-off principle:**
                        - ↑ γ → Better long-term planning but slower learning
                        - ↓ γ → Faster learning but more short-sighted
                        """)
                
                if os.path.exists('output/plots/gamma_sensitivity_conv.png'):
                    st.markdown("### Convergence Episodes vs Gamma (Detailed)")
                    st.image('output/plots/gamma_sensitivity_conv.png')
            
            # Display GIF if created
            if save_gif and os.path.exists(gif_path):
                st.markdown("---")
                st.subheader("🎬 Training Animation")
                
                # Create legend
                st.markdown("### Legend")
                cols_legend = st.columns(4)
                with cols_legend[0]:
                    st.markdown(f"<div style='background-color: {agent_color}; padding: 10px; border-radius: 5px; text-align: center;'>Main Agent</div>", 
                               unsafe_allow_html=True)
                with cols_legend[1]:
                    st.markdown(f"<div style='background-color: {goal_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;'>Goal</div>", 
                               unsafe_allow_html=True)
                with cols_legend[2]:
                    st.markdown(f"<div style='background-color: {obstacle_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;'>Obstacle</div>", 
                               unsafe_allow_html=True)
                with cols_legend[3]:
                    st.markdown(f"<div style='background-color: {other_agent_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;'>Other Agents</div>", 
                               unsafe_allow_html=True)
                
                st.markdown("**Watch** how the agent learns to navigate from top-left to the goal while avoiding obstacles!")
                with open(gif_path, 'rb') as f:
                    st.image(f.read())
            
            # Display report
            if generate_report_opt and os.path.exists(report_path):
                st.markdown("---")
                st.subheader("📄 Training Report")
                with open(report_path, 'r') as f:
                    report_content = f.read()
                
                with st.expander("📋 View Full Report", expanded=False):
                    st.text(report_content)
                
                st.download_button(
                    label="📥 Download Report", 
                    data=report_content,
                    file_name=os.path.basename(report_path),
                    mime='text/plain'
                )
        
        except Exception as e:
            st.error(f"❌ Training failed!")
            st.error(f"Error: {str(e)}")
            with st.expander("🔍 Full Error Traceback"):
                st.code(traceback.format_exc())

if __name__ == '__main__':
    main()