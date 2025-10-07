# Save this file as: enhanced_main.py

import argparse
from Livrable_3_2_WIAME_EL_HAFID import (
    convergence_vs_grid,
    visualize_gamma_sensitivity,
    GridWorldEnv,
    QLearningAgent,
    run_live_training,
    export_training_gif,
)
import os


def main():
    parser = argparse.ArgumentParser(description='Run experiments or live visualization')
    parser.add_argument('--mode', choices=['experiments', 'live', 'gif'], default='experiments', help='what to run')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--export-gif', action='store_true', help='Export training as GIF')
    parser.add_argument('--gif-episodes', type=int, default=50, help='Number of episodes for GIF')
    args = parser.parse_args()

    # Create output directory for models and GIFs
    os.makedirs('output', exist_ok=True)

    if args.mode == 'experiments':
        print("=" * 60)
        print("Running Convergence vs Grid Size Experiment")
        print("=" * 60)
        
        # Produce the requested summary plots and save them to disk.
        grid_sizes = [3, 5, 7, 9]
        obstacle_rel_positions = [(1/4, 2/5), (2/5, 3/5)]
        convergence_vs_grid(grid_sizes, base_goal_rel=(1.0, 1.0), 
                          obstacle_rel_positions=obstacle_rel_positions, 
                          episodes=100, n_runs=3, conv_tol=1e-4)

        print("\n" + "=" * 60)
        print("Running Gamma Sensitivity Experiment")
        print("=" * 60)
        
        # Gamma sensitivity with model saving
        rows = cols = 5
        goal_pos = [[rows-1, cols-1]]
        obstacles_pos = [[1,2],[2,3]]
        
        def factory():
            return GridWorldEnv(rows=rows, cols=cols, n_goals=1, goals_pos=goal_pos, 
                              goals_dynamic=False, n_obstacles=len(obstacles_pos), 
                              obstacles_pos=obstacles_pos, obstacles_dynamic=False,
                              n_other_agents=0, render=False, seed=None)
        
        gammas = [0.0, 0.3, 0.6, 0.9, 0.99]
        
        # If model saving is requested, train and save models for each gamma
        if args.save_models:
            print("\n" + "=" * 60)
            print("Training and Saving Models for Different Gamma Values")
            print("=" * 60)
            
            for g in gammas:
                print(f"\nTraining agent with gamma={g}...")
                env = factory()
                agent = QLearningAgent(env, episodes=150, gamma=g)
                agent.train()
                model_path = f'output/model_gamma_{g:.2f}'
                agent.save_model(model_path)
                env.close()
        
        # Run the visualization
        visualize_gamma_sensitivity(factory, gammas, episodes=150)
        
        # Export GIF for the standard configuration if requested
        if args.export_gif:
            print("\n" + "=" * 60)
            print("Exporting Training GIF (Standard Configuration)")
            print("=" * 60)
            
            env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], 
                             n_obstacles=2, obstacles_pos=[[1,2],[2,3]], 
                             render=False, seed=123)
            agent = QLearningAgent(env, episodes=args.gif_episodes)
            export_training_gif(env, agent, episodes=args.gif_episodes, 
                              out_path='output/training_standard.gif', fps=5)
            env.close()
            print(f"GIF saved to output/training_standard.gif")

    elif args.mode == 'live':
        print("=" * 60)
        print("Starting Live Visualization")
        print("=" * 60)
        
        # Attempt to run interactive visualization
        success = False
        try:
            run_live_training(rows=5, cols=5, goal_pos=[[4,4]], 
                            obstacles_pos=[[1,2],[2,3]], 
                            episodes=args.episodes, delay=0.05)
            success = True
        except Exception as e:
            print(f'\nLive visualization failed: {e}')
            print('Falling back to GIF export...')
            
            env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], 
                             n_obstacles=2, obstacles_pos=[[1,2],[2,3]], 
                             render=False, seed=123)
            agent = QLearningAgent(env, episodes=args.episodes)
            export_training_gif(env, agent, episodes=min(50, args.episodes), 
                              out_path='output/training_fallback.gif', fps=5)
            env.close()

        # Save the trained model if requested
        if args.save_models and success:
            print("\n" + "=" * 60)
            print("Saving Trained Model from Live Session")
            print("=" * 60)
            
            env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], 
                             n_obstacles=2, obstacles_pos=[[1,2],[2,3]], 
                             render=False, seed=123)
            agent = QLearningAgent(env, episodes=args.episodes)
            agent.train()
            agent.save_model('output/model_live_trained')
            env.close()

        # Run summary experiments
        print("\n" + "=" * 60)
        print("Running Summary Experiments")
        print("=" * 60)
        
        grid_sizes = [3, 5, 7, 9]
        obstacle_rel_positions = [(1/4, 2/5), (2/5, 3/5)]
        convergence_vs_grid(grid_sizes, base_goal_rel=(1.0, 1.0), 
                          obstacle_rel_positions=obstacle_rel_positions, 
                          episodes=100, n_runs=3, conv_tol=1e-4)
        
        rows = cols = 5
        goal_pos = [[rows-1, cols-1]]
        obstacles_pos = [[1,2],[2,3]]
        
        def factory():
            return GridWorldEnv(rows=rows, cols=cols, n_goals=1, goals_pos=goal_pos, 
                              goals_dynamic=False, n_obstacles=len(obstacles_pos), 
                              obstacles_pos=obstacles_pos, obstacles_dynamic=False,
                              n_other_agents=0, render=False, seed=None)
        
        gammas = [0.0, 0.3, 0.6, 0.9, 0.99]
        visualize_gamma_sensitivity(factory, gammas, episodes=150)
        
    else:  # gif mode
        print("=" * 60)
        print("Exporting Training GIF (Headless Mode)")
        print("=" * 60)
        
        env = GridWorldEnv(rows=5, cols=5, n_goals=1, goals_pos=[[4,4]], 
                         n_obstacles=2, obstacles_pos=[[1,2],[2,3]], 
                         render=False, seed=123)
        agent = QLearningAgent(env, episodes=args.episodes)
        
        gif_episodes = min(args.gif_episodes, args.episodes)
        export_training_gif(env, agent, episodes=gif_episodes, 
                          out_path='output/training_vis.gif', fps=5)
        
        # Save the trained model if requested
        if args.save_models:
            print("\nSaving trained model...")
            agent.save_model('output/model_gif_trained')
        
        env.close()
        print(f"\nGIF saved to output/training_vis.gif ({gif_episodes} episodes)")

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
    print("\nOutput files:")
    if os.path.exists('output'):
        for f in os.listdir('output'):
            print(f"  - output/{f}")


if __name__ == '__main__':
    main()