import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random




class WellnessJourneyEnv(gym.Env):
    
    """
    A mental health journey simulation environment where a patient navigates through 
    various mental health challenges to reach a wellness goal. The environment 
    represents a therapeutic landscape where:
    - The patient starts from an initial position
    - Red squares represent mental health challenges (anxiety, depression, etc.)
    - The green square represents the final wellness goal
    - Movement choices influence the patient's wellbeing points
    - Encountering challenges may cause setbacks but offers learning opportunities
    
    """ 
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}


    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size
        self.window_size = 600
        self.max_steps = 150


        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(4,), dtype=int)
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # down
        }


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None


        self.obstacles = []
        self.wellness_goal = None
        self.current_wellbeing_points = 0
        self.starting_point = np.array([0, 0])
        self.steps_taken = 0


    def _get_obs(self):
        return np.concatenate([self._patient_location, self._wellness_goal])


    def _get_info(self):
        return {"distance": np.linalg.norm(self._patient_location - self._wellness_goal, ord=1)}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._patient_location = self.starting_point.copy()
        self._wellness_goal = np.array([4, 4]) 


        # Defining obstacles as different mental health challenges
        self.obstacles = [
            (np.array([0, 2]), 'Anxiety'),
            (np.array([3, 2]), 'Negative Thoughts'),
            (np.array([3, 4]), 'Burnout'),
            (np.array([4, 1]), 'Panic Attack'),
            (np.array([1, 4]), 'Lack of Support'),
        ]


        self.current_wellbeing_points = 0
        self.steps_taken = 0


        observation = self._get_obs()
        info = self._get_info()


        if self.render_mode == "human":
            self._render_frame()


        return observation, info


    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = self._patient_location + direction


        # Check if the new location is within bounds
        if 0 <= new_location[0] < self.size and 0 <= new_location[1] < self.size:
            # Check if the new location is an obstacle (i.e., a mental health challenge)
            if new_location.tolist() not in [o[0].tolist() for o in self.obstacles]:
                self._patient_location = new_location
                reward = 1  # Small reward for making progress
            else:
                reward = -5  # Penalty for encountering a mental health challenge
                self._patient_location = random.choice([
                    np.array([x, y]) for x in range(self.size) for y in range(self.size)
                    if [x, y] not in [o[0].tolist() for o in self.obstacles] and not np.array_equal([x, y], self._patient_location)
                ])  # Teleport to a random valid location to avoid exploitation
        else:
            reward = -1  # Penalty for trying to move out of bounds


        # Add a small penalty for each step to discourage looping
        reward -= 0.1


        # Check if the patient has reached the wellness goal
        terminated = np.array_equal(self._patient_location, self._wellness_goal)
        if terminated:
            reward = 50  # Reward for reaching the wellness goal


        self.current_wellbeing_points += reward
        self.steps_taken += 1


        truncated = self.steps_taken >= self.max_steps
        done = terminated or truncated


        observation = self._get_obs()
        info = self._get_info()


        if self.render_mode == "human":
            self._render_frame()


        return observation, reward, done, truncated, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()


        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size


        # Draw obstacles (mental health challenges)
        for obstacle, label in self.obstacles:
            pygame.draw.rect(
                canvas,
                (128, 0, 0),
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                ),
            )
            font = pygame.font.SysFont(None, 24)
            text = font.render(label, True, (255, 255, 255))
            canvas.blit(text, (pix_square_size *
                        obstacle[0] + 5, pix_square_size * obstacle[1] + 5))


        # Draw the wellness goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._wellness_goal,
                (pix_square_size, pix_square_size),
            ),
        )
        font = pygame.font.SysFont(None, 24)
        text = font.render('Wellness Goal', True, (0, 0, 0))
        canvas.blit(text, (pix_square_size *
                    self._wellness_goal[0] + 5, pix_square_size * self._wellness_goal[1] + 5))


        # Draw the patient
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._patient_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        font = pygame.font.SysFont(None, 16)
        text = font.render('Patient', True, (255, 255, 255))
        canvas.blit(text, ((self._patient_location + 0.5) * pix_square_size - (pix_square_size / 6),
                    (self._patient_location + 0.5) * pix_square_size - (pix_square_size / 6)))


        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )


        # Draw current wellbeing points
        font = pygame.font.SysFont(None, 20)
        reward_text = font.render(
            f'Wellbeing Points: {self.current_wellbeing_points}', True, (0, 0, 0))
        canvas.blit(reward_text, (5, self.window_size - 25))


        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()