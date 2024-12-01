import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wellness_journey_env import WellnessJourneyEnv

def wrap_env():
    env = WellnessJourneyEnv()
    env = Monitor(env)
    return DummyVecEnv([lambda: env])

def train_agent():
    env = wrap_env()
    eval_env = wrap_env()
    
    model = DQN(
        "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
        env,
        learning_rate=2.5e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        progress_bar=True
    )
    
    model.save("policy")
    print("Training completed!")

if __name__ == "__main__":
    train_agent()
