# Mental Health Journey Simulation using Reinforcement Learning

## Project Overview
My mission is to address mental health issues, focusing on accessibility and awareness. This project stems from recognizing that most people lack understanding about mental health challenges and treatment paths. Through this simulation, I aim to make mental health journeys more visible and understandable.This project implements a reinforcement learning simulation modeling a patient's journey through mental health treatment. Using a Deep Q-Network (DQN) agent, it demonstrates how AI can learn to navigate therapeutic challenges, mirroring real-world mental health recovery processes.

## Features
- **Custom Gymnasium Environment**: 5x5 grid representing therapeutic landscape
- **Mental Health Challenges**: 
  - Anxiety 
  - Depression 
  - Trauma 
  - Social Anxiety 
  - Burnout 
- **Dynamic Wellness System**: Tracks patient's mental health status (0-100)
- **Visual Interface**: Real-time visualization of the agent's learning process
- **Reward Structure**: Simulates therapeutic progress and setbacks

## Technical Requirements
```
python 3.8+
gymnasium
stable-baselines3
pygame
numpy
tensorflow
```

## Installation
```bash
# Clone repository
git clone [https://github.com/rumunyana/mental_ai_DQN]

# Create virtual environment
python -m venv myenv

# Activate environment
# Windows:
myenv\Scripts\activate
# Unix/MacOS:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
mental-health-journey/
├── __pycache__/
├── best_model/
├── logs/
├── play.py                  # Simulation visualization script
├── policy.zip               # Saved policy file
├── README.md               # Documentation
├── test_env.py             # Environment testing script
├── train.py                # Training script
├── Wellness_dqn.zip        # Trained model
└── wellness_journey_env.py  # Environment implementation
```

## Usage
```bash
# Train the agent
python train.py

# Run simulation with trained model
python play.py
```

## Environment Details

**State Space:**
- Patient position (x, y)
- Goal position (x, y)
- Current wellness level

**Action Space:**
- Four directional movements (up, down, left, right)

## Visual Elements
- Blue Circle: Patient
- Red Squares: Mental Health Challenges
- Green Square: Wellness Goal
- Progress Bar: Current Wellness Level

## Acknowledgments
This project demonstrates the application of reinforcement learning in mental health contexts, focusing on:
- Path optimization through challenges
- Adaptive learning processes
- Visual representation of therapeutic journeys


## Link to video
https://youtu.be/bLv3e5F410E
link to the folder: https://drive.google.com/drive/folders/1ohtISuLEDHcepP-gw3EYopvw7vSlf9rt?usp=sharing