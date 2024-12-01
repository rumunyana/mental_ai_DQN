from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wellness_journey_env import WellnessJourneyEnv
import time
import tensorflow as tf

def train_wellness_agent():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    # Initialize the environment with no rendering during training
    env = DummyVecEnv([lambda: WellnessJourneyEnv(render_mode=None, size=5)])
    print("Success")


    # Define the model
    model = DQN(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.87,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        max_grad_norm=10,
    )


    # Training parameters
    total_episodes = 100  # Number of episodes to train


    # Train the model
    model.learn(total_timesteps=total_episodes * 5000)
    print("Training completed.")


    # Save the model
    model.save("Wellness_dqn.zip")


    # Simulate to check if the agent reaches the goal
    # Reinitialize
    env = DummyVecEnv([lambda: WellnessJourneyEnv(render_mode=None, size=5)])
    obs = env.reset()
    done = False
    goal_reached = False


    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            goal_reached = True
            break


    # Render the simulation if the goal was reached
    if goal_reached:
        print("Goal reached! Displaying the simulation...")
        # Reinitialize with rendering
        env = DummyVecEnv(
            [lambda: WellnessJourneyEnv(render_mode="human", size=5)])
        obs = env.reset()
        done = False


        # Render the environment step by step
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            env.render()
            # Adjust speed as needed
            time.sleep(1 / env.metadata["render_fps"])
    else:
        print("Failed.")


    # Close the environment
    env.close()




if __name__ == "__main__":
    train_wellness_agent()
