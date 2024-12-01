from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wellness_journey_env import WellnessJourneyEnv
import time


def play():
    # Initialize the environment with no rendering
    env = DummyVecEnv([lambda: WellnessJourneyEnv(size=5, render_mode=None)])

    # Load the trained model
    model = DQN.load("Wellness_dqn")

    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    max_steps = 100  

    for step in range(max_steps):
        if done:
            break  

        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Accumulate reward
        total_reward += reward[0]

        # Print progress for debugging
        print(f"Step: {step}, Obs: {obs}, Reward: {reward}, Done: {done}")

    # Render the simulation if the goal is reached
    if done:
        env = DummyVecEnv(
            [lambda: WellnessJourneyEnv(size=5, render_mode="human")])

        # Reset and re-run the simulation for visualization
        obs = env.reset()
        done = False
        for step in range(max_steps):
            if done:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(1 / env.metadata["render_fps"])

        print("Successfully got there.")
    else:
        print("Goal was not reached within the allowed steps.")

    # Print results
    print(f"Total Reward: {total_reward}")
    print(f"Goal Reached: {done}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    play()
