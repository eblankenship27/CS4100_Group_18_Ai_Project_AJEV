import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
import vis_overcooked_gym as vis

from Overcooked_sim_gym import OvercookedSimEnv

# Flags
train_flag  = 'train'  in sys.argv
render_flag = 'render' in sys.argv
phase2_flag = 'phase2' in sys.argv

if render_flag and phase2_flag:
    vis.setup(render=True, max_steps=1000, orders_to_complete=2)
    env = vis.game
elif render_flag:
    vis.setup(render=True)
    env = vis.game
elif phase2_flag:
    env = OvercookedSimEnv(render_mode=None, max_steps=1000, orders_to_complete=2)
else:
    env = OvercookedSimEnv(render_mode=None)

def hash_obs(obs):
    o = obs

    x = int(o[0] * 12)
    y = int(o[1] * 8)

    held = int(np.argmax(o[2:8]))
    facing = int(np.argmax(o[18:22]))

    # obs[8-9] = first plate, obs[10-11] = first fish
    plate_x = int(o[8] * 12) if o[8] >= 0 else -1
    plate_y = int(o[9] * 8) if o[9] >= 0 else -1

    fish_x = int(o[10] * 12) if o[10] >= 0 else -1
    fish_y = int(o[11] * 8) if o[11] >= 0 else -1
    
    shrimp_x = int(o[12] * 12) if o[12] >= 0 else -1
    shrimp_y = int(o[13] * 8) if o[13] >= 0 else -1
    
    cutFish_x = int(o[14] * 12) if o[14] >= 0 else -1
    cutFish_y = int(o[15] * 8) if o[15] >= 0 else -1
    
    cutShrimp_x = int(o[16] * 12) if o[16] >= 0 else -1
    cutShrimp_y = int(o[17] * 8) if o[17] >= 0 else -1
    

    # obs[22-23] = order one-hot (0=cutFish, 1=cutShrimp)
    # If both are negative the order is undetected (mdp_gym only) → use -1
    # to avoid aliasing with cutFish.
    order_slice = o[22:24]
    order = int(np.argmax(order_slice)) if np.max(order_slice) > 0 else -1

    # obs[24-29] = plate ingredient one-hot
    plate_ing = int(np.argmax(o[24:30]))

    return x, y, held, facing, plate_x, plate_y, fish_x, fish_y, shrimp_x, shrimp_y, cutFish_x, cutFish_y, cutShrimp_x, cutShrimp_y, order, plate_ing


# Q-Learning
def Q_learning(num_episodes=5000, gamma=0.99, epsilon=1.0, decay_rate=0.995,
               init_Q=None, init_updates=None):

    Q_table = dict(init_Q) if init_Q is not None else {}
    num_updates = dict(init_updates) if init_updates is not None else {}

    num_actions = env.action_space.n
    rewards_per_episode = []

    for episode in tqdm(range(num_episodes)):

        obs, _ = env.reset()
        state = hash_obs(obs)

        if state not in Q_table:
            Q_table[state] = np.zeros(num_actions)
            num_updates[state] = np.zeros(num_actions)

        total_reward = 0

        for t in range(env.max_steps):

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

            best_next = 0.0 if terminated else np.max(Q_table[next_state])
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

    return Q_table, num_updates, epsilon, rewards_per_episode


# Parameters
'''
Purpose	    num_episodes	decay_rate	Final ε	Notes
Smoke test	5_000	        0.9994	    ~0.05	Just verify it learns at all
Light run	50_000	        0.99994	    ~0.05	Usable policy, fast
Solid run	200_000	        0.999985	~0.05	Good balance — recommended starting point
Full training500_000	    0.999994	~0.05	Strong policy
Max	        1_000_000	    0.999997	~0.05	Diminishing returns past here
'''

# Phase 1 filenames (source for phase 2 curriculum seed)
PHASE1_TABLE   = "Q_table_1000000_0.999997.pickle"
PHASE1_UPDATES = "update_table_1000000_0.999997.pickle"

if phase2_flag:
    num_episodes = 500000
    decay_rate   = 0.999988
    gamma        = 0.99
    filename        = f"Q_table_phase2_{num_episodes}_{decay_rate}.pickle"
    update_filename = f"update_table_phase2_{num_episodes}_{decay_rate}.pickle"
else:
    num_episodes = 1000000
    decay_rate   = 0.999997
    gamma        = 0.9
    filename        = f"Q_table_{num_episodes}_{decay_rate}.pickle"
    update_filename = f"update_table_{num_episodes}_{decay_rate}.pickle"

def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)

# Training Mode
if train_flag:

    # Phase 2: seed from the phase 1 Q-table, then check for a phase 2 checkpoint
    if phase2_flag:
        init_Q, init_updates, init_epsilon = None, None, 1

        # Load phase 1 table as the starting point
        if os.path.exists(PHASE1_TABLE) and os.path.exists(PHASE1_UPDATES):
            with open(PHASE1_TABLE, "rb") as f:
                init_Q = pickle.load(f)
            with open(PHASE1_UPDATES, "rb") as f:
                checkpoint = pickle.load(f)
                init_updates = checkpoint["updates"]
            init_epsilon = 0.3  # warm start — preserve phase 1 knowledge
            print(f"Phase 2: seeded from {PHASE1_TABLE} ({len(init_Q)} states)\n")
        else:
            print(f"WARNING: phase 1 table not found at {PHASE1_TABLE}, starting from scratch\n")
            

        # If a phase 2 checkpoint already exists, resume from it instead
        if os.path.exists(filename) and os.path.exists(update_filename):
            with open(filename, "rb") as f:
                init_Q = pickle.load(f)
            with open(update_filename, "rb") as f:
                checkpoint = pickle.load(f)
                init_updates = checkpoint["updates"]
                init_epsilon = checkpoint.get("epsilon", 0.3)
            print(f"Resuming phase 2 from {filename} ({len(init_Q)} states, epsilon={init_epsilon:.4f})\n")

    # Phase 1: resume from checkpoint if one exists
    else:
        init_Q, init_updates, init_epsilon = None, None, 1.0
        if os.path.exists(filename) and os.path.exists(update_filename):
            with open(filename, "rb") as f:
                init_Q = pickle.load(f)
            with open(update_filename, "rb") as f:
                checkpoint = pickle.load(f)
                init_updates = checkpoint["updates"]
                init_epsilon = checkpoint.get("epsilon", 1.0)
            print(f"Resuming from {filename} ({len(init_Q)} states known, epsilon={init_epsilon:.4f})\n")
            
    init_epsilon = max(init_epsilon, 0.3)  # Don't let epsilon get too low on resume

    print("\nStarting training...\n")

    Q_table, num_updates, final_epsilon, rewards = Q_learning(
        num_episodes=num_episodes,
        gamma=gamma,
        epsilon=init_epsilon,
        decay_rate=decay_rate,
        init_Q=init_Q,
        init_updates=init_updates,
    )

    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)
    with open(update_filename, "wb") as f:
        pickle.dump({"updates": num_updates, "epsilon": final_epsilon}, f)

    print(f"\nSaved Q-table to {filename}")


# Evaluation Mode
else:

    if not os.path.exists(filename):
        print(f"No saved Q-table found at {filename}. Run with 'train' to train first.")
        sys.exit(1)

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
                action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))  # Select action using softmax over Q-values
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = hash_obs(obs)

            if render_flag:
                vis.refresh(obs, reward, terminated, truncated,
                            {'action': action}, delay=0.05)

    print("\nEvaluation Results:")
    print("Average reward:", total_reward / 1000)
    print("Total steps:", total_steps)
