from collections import defaultdict

import numpy as np
import gymnasium as gym
import pickle
import os


def hash_obs(obs):
    """Discretize a 30-float observation into a compact tuple state key.

    Identical to the hash_obs used in train_script.py so that Q-tables
    produced by either training path share the same state representation
    and can be loaded interchangeably.
    """
    o = obs

    x = int(o[0] * 12)
    y = int(o[1] * 8)

    held   = int(np.argmax(o[2:8]))
    facing = int(np.argmax(o[18:22]))

    # obs[8-9]   = first plate position
    # obs[10-11] = first fish position
    # obs[12-13] = first shrimp position
    # obs[14-15] = first cutFish position
    # obs[16-17] = first cutShrimp position
    plate_x    = int(o[8]  * 12) if o[8]  >= 0 else -1
    plate_y    = int(o[9]  * 8)  if o[9]  >= 0 else -1
    fish_x     = int(o[10] * 12) if o[10] >= 0 else -1
    fish_y     = int(o[11] * 8)  if o[11] >= 0 else -1
    shrimp_x   = int(o[12] * 12) if o[12] >= 0 else -1
    shrimp_y   = int(o[13] * 8)  if o[13] >= 0 else -1
    cutFish_x  = int(o[14] * 12) if o[14] >= 0 else -1
    cutFish_y  = int(o[15] * 8)  if o[15] >= 0 else -1
    cutShrimp_x = int(o[16] * 12) if o[16] >= 0 else -1
    cutShrimp_y = int(o[17] * 8)  if o[17] >= 0 else -1

    # obs[22-23]: order one-hot (index 0=cutFish, 1=cutShrimp)
    # If both are negative the order is undetected (mdp_gym only) → use -1
    # to avoid aliasing with cutFish.
    order_slice = o[22:24]
    order = int(np.argmax(order_slice)) if np.max(order_slice) > 0 else -1

    # obs[24-29]: plate ingredient one-hot (same 6-way encoding as held item)
    plate_ing = int(np.argmax(o[24:30]))

    return (x, y, held, facing,
            plate_x, plate_y, fish_x, fish_y,
            shrimp_x, shrimp_y, cutFish_x, cutFish_y, cutShrimp_x, cutShrimp_y,
            order, plate_ing)


class OvercookedAgent:
    def __init__(self, env: gym.Env, initial_epsilon: float, decay_rate: float, final_epsilon: float, discount_factor: float = 0.99):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        # Track number of updates for each (state, action)
        self.update_counts = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Learning factors
        self.discount_factor = discount_factor  # Discount factor
        
        # Exploration factors
        self.epsilon = initial_epsilon  # Exploration rate
        self.epsilon_decay = decay_rate
        self.final_epsilon = final_epsilon
        
    def select_action(self, state):
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, state, action: int, reward: float, terminated:bool, next_state):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # Q-learning update rule (off-policy)
        
        future_q_value = 0 if terminated else np.max(self.q_table[next_state])
        
        # Weight for weighted average
        weight = 1.0 / (1 + self.update_counts[state][action])
        td_target = reward + self.discount_factor * future_q_value
        # Convex combination update
        old_w_avg = (1 - weight) * self.q_table[state][action]
        new_w_value = weight * td_target
        
        # Set q-value for the current state, action to new weighted average
        self.q_table[state][action] = old_w_avg + new_w_value
        
        # Increment update count for this (state, action)
        self.update_counts[state][action] += 1
        
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename, update_filename):
        """Merges current Q-values into an existing save file, or creates one if none exists."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                saved = pickle.load(f)
            saved.update(dict(self.q_table))
        else:
            saved = dict(self.q_table)
        with open(filename, 'wb') as f:
            pickle.dump(saved, f)

        if os.path.exists(update_filename):
            with open(update_filename, 'rb') as f:
                u_saved = pickle.load(f)
            u_saved.update(dict(self.update_counts))
        else:
            u_saved = dict(self.update_counts)
        with open(update_filename, 'wb') as f:
            pickle.dump(u_saved, f)

    def load(self, q_filename, update_filename):
        """Loads a previously saved Q-table from a file."""
        with open(q_filename, 'rb') as f:
            saved = pickle.load(f)
        with open(update_filename, 'rb') as f:
            saved_updates = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), saved)
        self.update_counts = defaultdict(lambda: np.zeros(self.env.action_space.n), saved_updates)


def run_training(
    n_episodes: int = 1000,
    save_file: str = "q_table.pickle",
    update_file: str = "update_table.pickle",
    save_every: int = 50,
    render: bool = False,
    env_type: str = 'sim',
):
    """Train the Q-learning agent on Overcooked.

    Args:
        n_episodes:  Number of episodes to run.
        save_file:   Path to persist the Q-table (merged each save_every episodes).
        save_every:  How often (in episodes) to checkpoint the Q-table.
        render:      Whether to display the live game window during training.
    """

    render_mode = "human" if render else None
    if env_type == 'sim':
        from Overcooked_sim_gym import OvercookedSimEnv
        env = OvercookedSimEnv(render_mode=render_mode)
    else:
        from Overcooked_mdp_gym import OvercookedEnv
        env = OvercookedEnv(render_mode=render_mode)

    agent = OvercookedAgent(
        env=env,
        initial_epsilon=1.0,
        decay_rate=0.999,
        final_epsilon=0.1,
        discount_factor=0.99,
    )

    # Resume from checkpoint if one exists
    if os.path.exists(save_file) and os.path.exists(update_file):
        agent.load(save_file, update_filename=update_file)
        print(f"Loaded existing Q-table from {save_file}")

    episode_rewards = []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        state_id = hash_obs(obs)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state_id)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state_id = hash_obs(next_obs)

            agent.update(state_id, action, reward, terminated, next_state_id)

            state_id = next_state_id
            total_reward += reward

        # Decay epsilon after each episode
        agent.epsilon = max(
            agent.final_epsilon,
            agent.epsilon * agent.epsilon_decay
        )

        episode_rewards.append(total_reward)

        print(f"Episode {episode:>5}/{n_episodes}  reward={total_reward:>8.2f}  epsilon={agent.epsilon:.3f}")

        if episode % save_every == 0:
            agent.save(save_file, update_filename=update_file)
            print(f"  -> Q-table saved to {save_file}")

    # Final save
    agent.save(save_file, update_filename=update_file)
    env.close()
    print(f"\nTraining complete. Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")


if __name__ == "__main__":
    # In order to run training, make sure that overcooked is open and visable on screen, then run
    # python Overcooked_Q_Agent.py
    
    run_training(
        n_episodes=1000, 
        save_file='q_table.pickle', 
        update_file='update_table.pickle', 
        save_every=50, 
        render=False,
        env_type='sim', 
    )