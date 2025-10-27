"""
GridWorld Q-Learning Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Livrable_3_2_WIAME_EL_HAFID import (
    GridWorldEnv,
    QLearningAgent,
    visualize_learning_process,
    export_training_gif,
    convergence_vs_grid,
    visualize_gamma_sensitivity,
)

# Page configuration
st.set_page_config(
    page_title="GridWorld Q-Learning",
    page_icon="🤖",
    layout="wide"
)

def create_folders():
    """Create necessary output folders"""
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/gifs', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

def generate_report(config, rewards, deltas, training_time):
    """Generate training report"""
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
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Episodes Completed: {len(rewards)}\n")
        if rewards:
            f.write(f"Average Reward: {np.mean(rewards):.4f}\n")
            f.write(f"Final Episode Reward: {rewards[-1]:.4f}\n")
            f.write(f"Best Episode Reward: {max(rewards):.4f}\n")
            f.write(f"Worst Episode Reward: {min(rewards):.4f}\n")
        if deltas:
            f.write(f"Final Q-Delta: {deltas[-1]:.6e}\n")
            f.write(f"Convergence: {'Yes' if deltas[-1] < 1e-4 else 'No'}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    return report_path

def main():
    st.title("🤖 GridWorld Q-Learning Training")
    st.markdown("Configure and train a Q-Learning agent in a customizable grid environment")
    
    create_folders()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment Configuration
        st.subheader("🌍 Environment")
        rows = st.number_input("Grid Rows", min_value=2, max_value=20, value=5)
        cols = st.number_input("Grid Columns", min_value=2, max_value=20, value=5)
        
        # Goals
        st.markdown("**Goals**")
        n_goals = st.number_input("Number of Goals", min_value=1, max_value=10, value=1)
        use_default_goal = st.checkbox("Place goal at bottom-right", value=True)
        
        if use_default_goal:
            goals_pos = [[rows-1, cols-1]]
        else:
            goals_pos = []
            for i in range(n_goals):
                col1, col2 = st.columns(2)
                with col1:
                    g_row = st.number_input(f"Goal {i+1} Row", 0, rows-1, min(i, rows-1), key=f"g_row_{i}")
                with col2:
                    g_col = st.number_input(f"Goal {i+1} Col", 0, cols-1, min(i, cols-1), key=f"g_col_{i}")
                goals_pos.append([g_row, g_col])
        
        goals_dynamic = st.checkbox("Goals move randomly", value=False)
        
        # Obstacles
        st.markdown("**Obstacles**")
        n_obstacles = st.number_input("Number of Obstacles", min_value=0, max_value=20, value=2)
        obstacles_pos = []
        if n_obstacles > 0:
            for i in range(n_obstacles):
                col1, col2 = st.columns(2)
                with col1:
                    o_row = st.number_input(f"Obstacle {i+1} Row", 0, rows-1, 
                                          min(1+i, rows-1), key=f"o_row_{i}")
                with col2:
                    o_col = st.number_input(f"Obstacle {i+1} Col", 0, cols-1, 
                                          min(2+i, cols-1), key=f"o_col_{i}")
                obstacles_pos.append([o_row, o_col])
            obstacles_dynamic = st.checkbox("Obstacles move randomly", value=False)
        else:
            obstacles_pos = None
            obstacles_dynamic = False
        
        # Other Agents
        st.markdown("**Other Agents**")
        n_other_agents = st.number_input("Number of Other Agents", min_value=0, max_value=10, value=0)
        other_agents_pos = []
        if n_other_agents > 0:
            for i in range(n_other_agents):
                col1, col2 = st.columns(2)
                with col1:
                    a_row = st.number_input(f"Agent {i+1} Row", 0, rows-1, 
                                          min(2+i, rows-1), key=f"a_row_{i}")
                with col2:
                    a_col = st.number_input(f"Agent {i+1} Col", 0, cols-1, 
                                          min(1+i, cols-1), key=f"a_col_{i}")
                other_agents_pos.append([a_row, a_col])
            other_agents_dynamic = st.checkbox("Other agents move randomly", value=False)
        else:
            other_agents_pos = None
            other_agents_dynamic = False
        
        # Training Configuration
        st.subheader("🎓 Training")
        episodes = st.number_input("Training Episodes", min_value=10, max_value=10000, value=200)
        
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
        
        generate_report_opt = st.checkbox("Generate training report", value=True)
        
        # Plot Options
        st.subheader("📊 Plots")
        plot_rewards = st.checkbox("Episode Rewards", value=True)
        plot_qdeltas = st.checkbox("Q-Delta Convergence", value=True)
        plot_combined = st.checkbox("Combined Plot", value=True)
        
        st.markdown("**Advanced Analysis**")
        run_grid_analysis = st.checkbox("Grid Size Analysis", value=False)
        run_gamma_analysis = st.checkbox("Gamma Sensitivity", value=False)
        
        # Additional gamma models
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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
        rewards, deltas, _ = agent.train_track(episodes=episodes)
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
            export_training_gif(gif_env, gif_agent, episodes=gif_episodes, 
                              out_path=gif_path, fps=gif_fps)
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
            
            # Train and save models for each gamma if requested
            if save_gamma_models:
                for g in gammas:
                    g_env = env_factory()
                    g_agent = QLearningAgent(g_env, episodes=min(150, episodes), gamma=g)
                    g_agent.train()
                    g_agent.save_model(f'output/model_gamma_{g:.2f}')
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
            report_path = generate_report(config, rewards, deltas, training_time)
        
        progress_bar.progress(1.0)
        status_text.text("✅ All tasks completed!")
        
        env.close()
        
        # Display results
        st.success("Training completed successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Episodes", len(rewards))
        with col2:
            st.metric("Avg Reward", f"{np.mean(rewards):.3f}" if rewards else "N/A")
        with col3:
            st.metric("Training Time", f"{training_time:.1f}s")
        
        # Display plots
        if plot_rewards and os.path.exists(f'output/plots/rewards_{rows}x{cols}.png'):
            st.image(f'output/plots/rewards_{rows}x{cols}.png', caption="Episode Rewards")
        
        if plot_qdeltas and os.path.exists(f'output/plots/qdeltas_{rows}x{cols}.png'):
            st.image(f'output/plots/qdeltas_{rows}x{cols}.png', caption="Q-Delta Convergence")
        
        if plot_combined and os.path.exists(f'output/plots/combined_{rows}x{cols}.png'):
            st.image(f'output/plots/combined_{rows}x{cols}.png', caption="Combined Convergence")
        
        # Display GIF if created
        if save_gif and os.path.exists(gif_path):
            st.subheader("Training Animation")
            with open(gif_path, 'rb') as f:
                st.image(f.read())
        
        # Display report
        if generate_report_opt and os.path.exists(report_path):
            st.subheader("Training Report")
            with open(report_path, 'r') as f:
                st.text(f.read())
            st.download_button("Download Report", open(report_path, 'r').read(), 
                             file_name=os.path.basename(report_path))

if __name__ == '__main__':
    main()