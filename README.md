# This is the GitHub Repository for CS4100 Artifical Intelligence Project 
### Group Members: Evan Blankenship, Jessie Pfahler, Adam Mayyalou, Vineel Bandla

- Link to Presentation: https://docs.google.com/presentation/d/1hhJSlkTCFfhiStJZhEhxpJGXmrtsQxjSWZJkrws2rmw/edit?usp=sharing

**Execution Instructions:**

**Approach Statement:**

This project is made up of 3 distinct parts, a visual pipeline, a gym enviornment, and a simulated game enviornment used for training.

The _vision pipeline_ is responsible for converting active gameplay and to a structured state repersentation used by the machine learning model. It uses OpenCV to capture real-time frames from the active game play and processes them through a trained YOLO model to gather location information on various objects on the screen. These objects include the chef, the ingredients, and the plates. Additionally, it uses the template matching to identify the static ticket and therefore the current order.

The code files and images used for training the YOLO model as well as other resources for the object recognition can be found in the **src/objectTrackingModels** folder.

The _real-world Gym enviornment_ is the wrapper used for the trained model to be able to play the real game. It interacts both with the real-game using the visual pipeline and executing actions using Pyautogui. This allows the agent to interact with the game in the real-enviornment while utilizing our trained model to make decisions.

This enviornment can be found within the file path **src/mdp/Overcooked_mdp_gym.py**.

The _simulated environment_ is a grid-based repersentation of the game that allowed for more efficient training. It uses the same state structure as the one visually derives from the real enviornment, developing policies in the simulation that can be directed translated tothe real gameplay. This enviornment is used to train the agent using Q-learning.

This simulated enviornment can be found in the files **src/mdp/Overcooked_sim_gym.py** and **src/mdp/vis_overcooked_gym.py**.

