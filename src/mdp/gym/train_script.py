import sys
import time
import pickle
import numpy as np
from tqdm import tqdm

from Overcooked_sim_gym import OvercookedSimEnv

# Flags
train_flag = 'train' in sys.argv
render_flag = 'render' in sys.argv

env = OvercookedSimEnv(render_mode="human" if render_flag else None)

def hash_obs(obs):
    o = obs

    x = int(o[0] * 5)
    y = int(o[1] * 5)

    held = np.argmax(o[2:8])
    facing = np.argmax(o[24:28])

    # Add positions of key objects
    fish_x = int(o[8] * 5) if o[8] >= 0 else -1
    fish_y = int(o[9] * 5) if o[9] >= 0 else -1

    plate_x = int(o[8 + 2*0] * 5) if o[8] >= 0 else -1
    plate_y = int(o[9 + 2*0] * 5) if o[9] >= 0 else -1

    return x, y, held, facing, fish_x, fish_y, plate_x, plate_y


# Q-Learning
def Q_learning(num_episodes=5000, gamma=0.9, epsilon=1.0, decay_rate=0.995):

    Q_table = {}
    num_updates = {}

    num_actions = env.action_space.n
    rewards_per_episode = []

    for episode in tqdm(range(num_episodes)):

        obs, _ = env.reset()
        state = hash_obs(obs)

        if state not in Q_table:
            Q_table[state] = np.zeros(num_actions)
            num_updates[state] = np.zeros(num_actions)

        total_reward = 0

        for t in range(500):

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = hash_obs(next_obs)

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(num_actions)
                num_updates[next_state] = np.zeros(num_actions)

            num_updates[state][action] += 1
            eta = 1 / (1 + num_updates[state][action])

            best_next = np.max(Q_table[next_state])
            Q_table[state][action] = (
                (1 - eta) * Q_table[state][action]
                + eta * (reward + gamma * best_next)
            )

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

        epsilon *= decay_rate

    return Q_table, rewards_per_episode


# Parameters
num_episodes = 1000000
decay_rate = 0.997

filename = f"Q_table_{num_episodes}_{decay_rate}.pickle"


# Training Mode
if train_flag:

    print("\nStarting training...\n")

    Q_table, rewards = Q_learning(
        num_episodes=num_episodes,
        gamma=0.9,
        epsilon=1.0,
        decay_rate=decay_rate
    )

    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)

    print(f"\nSaved Q-table to {filename}")


# Evaluation Mode
else:

    print(f"\nLoading {filename}...\n")

    with open(filename, "rb") as f:
        Q_table = pickle.load(f)

    total_reward = 0
    total_steps = 0

    for episode in tqdm(range(1000)):

        obs, _ = env.reset()
        state = hash_obs(obs)

        done = False

        while not done:
            total_steps += 1

            if state in Q_table:
                action = np.argmax(Q_table[state])
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = hash_obs(obs)

            if render_flag:
                time.sleep(0.05)

    print("\nEvaluation Results:")
    print("Average reward:", total_reward / 1000)
    print("Total steps:", total_steps)
