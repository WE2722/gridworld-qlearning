"""
GridWorld Q-Learning Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Add debug info
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.text(f"Python: {sys.version.split()[0]}")
st.sidebar.text(f"Working dir: {os.getcwd()}")
st.sidebar.text(f"Files: {os.listdir('.')}")

# Import with detailed error handling
try:
    from Livrable_3_2_WIAME_EL_HAFID import (
        GridWorldEnv,
        QLearningAgent,
        export_training_gif_custom_colors,
        convergence_vs_grid,
        visualize_gamma_sensitivity,
    )
    st.sidebar.success("✅ Imports successful")
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Available files in directory:")
    st.code("\n".join(os.listdir('.')))
    st.error("Make sure 'Livrable_3_2_WIAME_EL_HAFID.py' exists in your repository")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error during import: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Page configuration
st.set_page_config(
    page_title="GridWorld Q-Learning",
    page_icon="🤖",
    layout="wide"
)

def create_folders():
    """Create necessary output folders"""
    try:
        os.makedirs('output/plots', exist_ok=True)
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/gifs', exist_ok=True)
        os.makedirs('output/reports', exist_ok=True)
        return True
    except Exception as e:
        st.warning(f"Could not create output folders: {e}")
        return False

def generate_report(config, rewards, deltas, training_time, goal_reached_count):
    """Generate training report"""
    try:
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
            f.write(f"Goals: {config['n_goals']}\n")
            f.write(f"Obstacles: {config['n_obstacles']}\n")
            f.write(f"Episodes: {config['episodes']}\n\n")
            
            f.write("TRAINING RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Episodes Completed: {len(rewards)}\n")
            f.write(f"Goal Reached: {goal_reached_count} times\n")
            
            if rewards:
                f.write(f"Average Reward: {np.mean(rewards):.4f}\n")
                f.write(f"Best Reward: {max(rewards):.4f}\n")
        
        return report_path
    except Exception as e:
        st.error(f"Failed to generate report: {e}")
        return None

def main():
    st.title("🤖 GridWorld Q-Learning Training")
    st.markdown("Configure and train a Q-Learning agent")
    
    # Create folders
    folders_ok = create_folders()
    if not folders_ok:
        st.warning("⚠️ Output folders could not be created. Files may not be saved.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment
        st.subheader("🌍 Environment")
        rows = st.number_input("Rows", 2, 10, 5)
        cols = st.number_input("Cols", 2, 10, 5)
        
        # Goals
        st.markdown("**Goals**")
        n_goals = st.number_input("Number of Goals", 1, 5, 1)
        use_default_goal = st.checkbox("Bottom-right goal", value=True)
        
        if use_default_goal:
            goals_pos = [[rows-1, cols-1]]
        else:
            goals_pos = []
            for i in range(n_goals):
                col1, col2 = st.columns(2)
                with col1:
                    g_row = st.number_input(f"G{i+1} Row", 0, rows-1, min(i, rows-1), key=f"gr_{i}")
                with col2:
                    g_col = st.number_input(f"G{i+1} Col", 0, cols-1, min(i, cols-1), key=f"gc_{i}")
                goals_pos.append([g_row, g_col])
        
        goals_dynamic = st.checkbox("Dynamic goals", value=False)
        
        # Obstacles
        st.markdown("**Obstacles**")
        n_obstacles = st.number_input("Number of Obstacles", 0, 10, 2)
        obstacles_pos = []
        obstacles_dynamic = False
        
        if n_obstacles > 0:
            for i in range(n_obstacles):
                col1, col2 = st.columns(2)
                with col1:
                    o_row = st.number_input(f"O{i+1} Row", 0, rows-1, min(1+i, rows-1), key=f"or_{i}")
                with col2:
                    o_col = st.number_input(f"O{i+1} Col", 0, cols-1, min(2+i, cols-1), key=f"oc_{i}")
                obstacles_pos.append([o_row, o_col])
            obstacles_dynamic = st.checkbox("Dynamic obstacles", value=False)
        else:
            obstacles_pos = None
        
        # Other Agents
        st.markdown("**Other Agents**")
        n_other_agents = st.number_input("Other Agents", 0, 5, 0)
        other_agents_pos = []
        other_agents_dynamic = False
        
        if n_other_agents > 0:
            for i in range(n_other_agents):
                col1, col2 = st.columns(2)
                with col1:
                    a_row = st.number_input(f"A{i+1} Row", 0, rows-1, min(2+i, rows-1), key=f"ar_{i}")
                with col2:
                    a_col = st.number_input(f"A{i+1} Col", 0, cols-1, min(1+i, cols-1), key=f"ac_{i}")
                other_agents_pos.append([a_row, a_col])
            other_agents_dynamic = st.checkbox("Dynamic agents", value=False)
        else:
            other_agents_pos = None
        
        # Training
        st.subheader("🎓 Training")
        episodes = st.number_input("Episodes", 10, 1000, 50)
        
        use_defaults = st.checkbox("Default params", value=True)
        if use_defaults:
            alpha, gamma, epsilon = 0.1, 0.9, 0.1
        else:
            alpha = st.slider("Alpha", 0.0, 1.0, 0.1, 0.01)
            gamma = st.slider("Gamma", 0.0, 1.0, 0.9, 0.01)
            epsilon = st.slider("Epsilon", 0.0, 1.0, 0.1, 0.01)
        
        max_steps = max(100, rows * cols * 4)
        reward_shaping = st.checkbox("Reward shaping", value=False)
        
        # Outputs
        st.subheader("💾 Outputs")
        save_model = st.checkbox("Save model", value=False)
        save_gif = st.checkbox("Export GIF", value=False)
        
        if save_gif:
            gif_episodes = st.number_input("GIF Episodes", 1, episodes, min(10, episodes))
            gif_fps = st.number_input("FPS", 1, 30, 5)
            agent_color = st.color_picker("Agent", "#00FF00")
            goal_color = st.color_picker("Goal", "#FF0000")
            obstacle_color = st.color_picker("Obstacle", "#000000")
            other_agent_color = st.color_picker("Others", "#009600")
        
        generate_report_opt = st.checkbox("Generate report", value=False)
        
        # Plots
        st.subheader("📊 Plots")
        plot_rewards = st.checkbox("Rewards plot", value=True)
        plot_qdeltas = st.checkbox("Q-Delta plot", value=True)
        plot_combined = st.checkbox("Combined plot", value=False)
        
        # Advanced
        run_grid_analysis = st.checkbox("Grid analysis", value=False)
        run_gamma_analysis = st.checkbox("Gamma analysis", value=False)
        save_gamma_models = False
        if run_gamma_analysis:
            save_gamma_models = st.checkbox("Save gamma models", value=False)
    
    # Main content
    st.subheader("📋 Configuration")
    st.markdown(f"""
    **Grid:** {rows}×{cols}  
    **Goals:** {n_goals} {'(dynamic)' if goals_dynamic else '(static)'}  
    **Obstacles:** {n_obstacles} {'(dynamic)' if obstacles_dynamic else '(static)'}  
    **Episodes:** {episodes}  
    **Params:** α={alpha}, γ={gamma}, ε={epsilon}
    """)
    
    # Train button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        st.write("🔍 **Debug: Button clicked**")
        
        try:
            # Validation
            st.write("📝 Step 1: Validating positions...")
            all_positions = {(0, 0)}
            
            for g in goals_pos:
                pos = tuple(g)
                if pos in all_positions:
                    st.error(f"❌ Conflict at {pos}")
                    st.stop()
                all_positions.add(pos)
            
            if obstacles_pos:
                for o in obstacles_pos:
                    pos = tuple(o)
                    if pos in all_positions:
                        st.error(f"❌ Conflict at {pos}")
                        st.stop()
                    all_positions.add(pos)
            
            if other_agents_pos:
                for a in other_agents_pos:
                    pos = tuple(a)
                    if pos in all_positions:
                        st.error(f"❌ Conflict at {pos}")
                        st.stop()
                    all_positions.add(pos)
            
            st.success("✅ Positions validated")
            
            # Create environment
            st.write("🌍 Step 2: Creating environment...")
            env = GridWorldEnv(
                rows=rows, cols=cols,
                n_goals=n_goals, goals_pos=goals_pos, goals_dynamic=goals_dynamic,
                n_obstacles=n_obstacles, obstacles_pos=obstacles_pos, obstacles_dynamic=obstacles_dynamic,
                n_other_agents=n_other_agents, other_agents_pos=other_agents_pos,
                other_agents_dynamic=other_agents_dynamic,
                render=False, max_steps=max_steps, seed=123
            )
            st.success("✅ Environment created")
            
            # Create agent
            st.write("🤖 Step 3: Creating agent...")
            agent = QLearningAgent(env, episodes=episodes, alpha=alpha, gamma=gamma, 
                                 epsilon=epsilon, reward_shaping=reward_shaping)
            st.success("✅ Agent created")
            
            # Training
            st.write("🎓 Step 4: Training...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
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
                
                while not done and t < max_steps:
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
                
                if ep % max(1, episodes // 10) == 0:
                    progress = min(1.0, (ep + 1) / episodes)
                    progress_bar.progress(progress)
                    status_text.text(f"Episode {ep+1}/{episodes}")
            
            training_time = time.time() - start_time
            rewards = rewards_list
            deltas = deltas_list
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training complete!")
            
            env.close()
            
            # Results
            st.success(f"🎉 Training completed in {training_time:.1f}s!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Episodes", len(rewards))
            with col2:
                st.metric("Avg Reward", f"{np.mean(rewards):.3f}")
            with col3:
                st.metric("Goals", goal_reached_count)
            with col4:
                st.metric("Success", f"{goal_reached_count/len(rewards)*100:.1f}%")
            
            # Plot rewards
            if plot_rewards and len(rewards) > 0:
                st.subheader("📈 Episode Rewards")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(1, len(rewards)+1), rewards, '-', color='tab:blue', linewidth=1.5)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.set_title('Training Progress')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if folders_ok:
                    plt.savefig(f'output/plots/rewards_{rows}x{cols}.png', dpi=100)
                
                st.pyplot(fig)
                plt.close()
            
            # Plot Q-deltas
            if plot_qdeltas and len(deltas) > 0:
                st.subheader("📉 Q-Table Convergence")
                fig, ax = plt.subplots(figsize=(10, 5))
                safe_deltas = [max(d, 1e-12) for d in deltas]
                ax.plot(range(1, len(safe_deltas)+1), safe_deltas, '-', color='tab:orange', linewidth=1.5)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Max Q-Delta')
                ax.set_yscale('log')
                ax.set_title('Q-Table Stability')
                ax.grid(True, which='both', alpha=0.3)
                plt.tight_layout()
                
                if folders_ok:
                    plt.savefig(f'output/plots/qdeltas_{rows}x{cols}.png', dpi=100)
                
                st.pyplot(fig)
                plt.close()
            
            # Save model
            if save_model and folders_ok:
                model_path = f'output/models/model_{rows}x{cols}'
                agent.save_model(model_path)
                st.success(f"💾 Model saved to {model_path}")
            
            # Generate GIF
            if save_gif and folders_ok:
                try:
                    st.write("🎬 Generating GIF...")
                    gif_env = GridWorldEnv(
                        rows=rows, cols=cols, n_goals=n_goals, goals_pos=goals_pos, goals_dynamic=goals_dynamic,
                        n_obstacles=n_obstacles, obstacles_pos=obstacles_pos, obstacles_dynamic=obstacles_dynamic,
                        n_other_agents=n_other_agents, other_agents_pos=other_agents_pos,
                        other_agents_dynamic=other_agents_dynamic,
                        render=False, max_steps=max_steps, seed=123
                    )
                    gif_agent = QLearningAgent(gif_env, episodes=gif_episodes, alpha=alpha, 
                                              gamma=gamma, epsilon=epsilon, reward_shaping=reward_shaping)
                    gif_path = f'output/gifs/training.gif'
                    
                    colors = {
                        'agent': agent_color,
                        'goal': goal_color,
                        'obstacle': obstacle_color,
                        'other_agent': other_agent_color
                    }
                    
                    export_training_gif_custom_colors(gif_env, gif_agent, episodes=gif_episodes, 
                                                     out_path=gif_path, fps=gif_fps, colors=colors)
                    gif_env.close()
                    
                    if os.path.exists(gif_path):
                        st.success("✅ GIF created")
                        with open(gif_path, 'rb') as f:
                            st.image(f.read())
                except Exception as e:
                    st.warning(f"GIF generation failed: {e}")
            
            # Report
            if generate_report_opt and folders_ok:
                config = {
                    'rows': rows, 'cols': cols, 'n_goals': n_goals, 'n_obstacles': n_obstacles,
                    'n_other_agents': n_other_agents, 'goals_dynamic': goals_dynamic,
                    'obstacles_dynamic': obstacles_dynamic, 'other_agents_dynamic': other_agents_dynamic,
                    'episodes': episodes, 'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon,
                    'reward_shaping': reward_shaping, 'max_steps': max_steps, 'goals_pos': goals_pos,
                    'obstacles_pos': obstacles_pos, 'other_agents_pos': other_agents_pos
                }
                report_path = generate_report(config, rewards, deltas, training_time, goal_reached_count)
                if report_path and os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        st.download_button("📥 Download Report", f.read(), file_name="report.txt")
        
        except Exception as e:
            st.error(f"❌ Error occurred!")
            st.error(f"**Error type:** {type(e).__name__}")
            st.error(f"**Error message:** {str(e)}")
            with st.expander("🔍 Full Traceback"):
                st.code(traceback.format_exc())

if __name__ == '__main__':
    main()