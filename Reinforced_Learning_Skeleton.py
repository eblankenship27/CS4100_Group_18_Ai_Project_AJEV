import numpy as np
import random


class QLearningAgent:
	def __init__(self, state_space_size, action_space_size, gamma=0.99, epsilon=0.1):
		"""
		The initialization method sets up the Q-learning agent with the given attributes:
        - state_space_size: The number of possible states in the environment.
        - action_space_size: The number of possible actions the agent can take.
        - gamma: The discount factor for future rewards, which determines how much the agent values future rewards compared to immediate rewards.
        - epsilon: The exploration rate, which determines the probability of the agent choosing a random action (exploration) versus choosing the action with the highest Q-value (exploitation). A higher epsilon encourages more exploration, while a lower epsilon encourages more exploitation.
		"""
		self.state_space_size = state_space_size
		self.action_space_size = action_space_size
		self.gamma = gamma  # Discount factor
		self.epsilon = epsilon  # Exploration rate
		self.q_table = np.zeros((state_space_size, action_space_size))
		# Track number of updates for each (state, action)
		self.update_counts = np.zeros((state_space_size, action_space_size), dtype=int)

	def select_action(self, state):
		"""The select_action method implements the epsilon-greedy action selection strategy. It takes the current state as input and returns an action based on the following logic:
        - With probability epsilon, the agent selects a random action (exploration).
        - With probability (1 - epsilon), the agent selects the action with the highest Q-value for the current state (exploitation).
        This strategy allows the agent to balance exploration and exploitation, enabling it to learn effectively over time.
        """
		# Epsilon-greedy action selection
		if random.uniform(0, 1) < self.epsilon:
			return random.randint(0, self.action_space_size - 1)
		else:
			return np.argmax(self.q_table[state])

	def update(self, state, action, reward, next_state):
		"""
		The update method implements the Q-learning update rule, which is used to update the Q-values based on the agent's experience. The method takes the following parameters:
        - state: The current state of the environment before taking the action.
        - action: The action taken by the agent in the current state.
        - reward: The reward received after taking the action.
        - next_state: The state of the environment after taking the action.
		"""
		# Q-learning update rule (off-policy)
		
		best_next_action = np.argmax(self.q_table[next_state])
		V_opt = self.q_table[next_state, best_next_action]
		nu = 1.0 / (1 + self.update_counts[state, action])
		td_target = reward + self.gamma * V_opt
		# Convex combination update
		self.q_table[state, action] = (1 - nu) * self.q_table[state, action] + nu * td_target
		# Increment update count for this (state, action)
		self.update_counts[state, action] += 1

	def save(self, filename):
		"""The save method allows the agent to save its current Q-table to a file. This is useful for preserving the learned knowledge and for later analysis or deployment. The method takes the filename as input and saves the Q-table in a format that can be easily loaded later.
        """
		np.save(filename, self.q_table)

	def load(self, filename):
		"""The load method allows the agent to load a previously saved Q-table from a file. This is useful for resuming training or for using a pre-trained model. The method takes the filename as input and loads the Q-table into the agent's memory.
        """
		self.q_table = np.load(filename)
