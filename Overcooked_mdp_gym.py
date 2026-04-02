import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import pyautogui
import mss
from ultralytics import YOLO

CLASS_NAMES = {
    0: "chef",
    1: "plate",
    2: "fish",
    3: "shrimp",
    4: "cutFish",
    5: "cutShrimp",
    6: "servingCounter",
    7: "fishBox",
    8: "shrimpBox"
}

# Held-item indices (used for one-hot encoding in observation)
HOLD_NONE = 0
HOLD_PLATE = 1
HOLD_FISH = 2
HOLD_SHRIMP = 3
HOLD_CUTFISH = 4
HOLD_CUTSHRIMP = 5

# Action constants
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CHOP = 4
ACTION_PICKUP = 5

KEYMAP = {
    ACTION_UP: 'up',
    ACTION_DOWN: 'down',
    ACTION_LEFT: 'left',
    ACTION_RIGHT: 'right',
    ACTION_CHOP: 'x',
    ACTION_PICKUP: 'space'
}


class OvercookedEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        players: int = 1,
        render_mode: str = None,
        model_path: str = "objectTrackingModels/runs/detect/train2/weights/best.pt",
    ):
        super().__init__()
        self.players = players
        self.render_mode = render_mode

        # Set up the 
        self.model = YOLO(model_path)
        self.sct = mss.mss()
        self.monitor = {"top": 200, "left": 0, "width": 600, "height": 400}

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CHOP', 'PICKUP']
        self.action_space = spaces.Discrete(len(self.actions))

        # Continuous state observations hashed into 24-float observation vector:
        # [0-1]   chef (x, y)
        # [2-7]   one-hot: held item (none/plate/fish/shrimp/cutFish/cutShrimp)
        # [8-9]   first plate center
        # [10-11] first fish center
        # [12-13] first shrimp center
        # [14-15] first cutFish center
        # [16-17] first cutShrimp center
        # [18-19] serving counter center (from YOLO class 6)
        # [20-21] fish box center (from YOLO class 7)
        # [22-23] shrimp box center (from YOLO class 8)
        # Undetected objects default to (-1, -1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        self.max_steps = 500
        self.step_count = 0
        self.prev_state = None
        self.held_item = HOLD_NONE

    # ------------------------------------------------------------------
    # Vision pipeline
    # ------------------------------------------------------------------

    def _capture_frame(self):
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return frame

    def _detect_objects(self, frame):
        results = self.model(frame, conf=0.1, verbose=False)
        detections = []
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2 / self.monitor["width"]
                cy = (y1 + y2) / 2 / self.monitor["height"]
                detections.append({
                    "class": int(cls),
                    "class_name": CLASS_NAMES.get(int(cls), "unknown"),
                    "confidence": float(conf),
                    "center": (cx, cy)
                })
        return detections

    def _build_game_state(self, detections):
        state = {
            "chef": None,
            "plates": [],
            "fish": [],
            "shrimp": [],
            "cutFish": [],
            "cutShrimp": [],
            "servingCounter": None,
            "fishBox": None,
            "shrimpBox": None
        }
        class_to_key = {
            0: "chef",
            1: "plates",
            2: "fish",
            3: "shrimp",
            4: "cutFish",
            5: "cutShrimp",
            6: "servingCounter",
            7: "fishBox",
            8: "shrimpBox"
        }
        for det in detections:
            key = class_to_key.get(det["class"])
            if key is None:
                continue
            if key in ("chef", "servingCounter", "fishBox", "shrimpBox"):
                state[key] = det["center"]
            else:
                state[key].append(det["center"])
        return state

    def _encode_state(self, game_state):
        obs = np.full(24, -1.0, dtype=np.float32)

        if game_state["chef"] is not None:
            obs[0], obs[1] = game_state["chef"]

        obs[2 + self.held_item] = 1.0

        for i, key in enumerate(["plates", "fish", "shrimp", "cutFish", "cutShrimp"]):
            if game_state[key]:
                obs[8 + 2 * i] = game_state[key][0][0]
                obs[8 + 2 * i + 1] = game_state[key][0][1]

        for i, key in enumerate(["servingCounter", "fishBox", "shrimpBox"]):
            if game_state[key] is not None:
                obs[18 + 2 * i] = game_state[key][0]
                obs[18 + 2 * i + 1] = game_state[key][1]

        return obs

    # ------------------------------------------------------------------
    # Held-item tracking
    # ------------------------------------------------------------------

    def _nearest_item(self, chef_pos, item_positions, threshold=0.15):
        if chef_pos is None or not item_positions:
            return False
        cx, cy = chef_pos
        for pos in item_positions:
            dist = ((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2) ** 0.5
            if dist < threshold:
                return True
        return False

    def _update_held_item(self, action, prev_state):
        if action != ACTION_PICKUP:
            return
        if self.held_item != HOLD_NONE:
            self.held_item = HOLD_NONE
            return
        chef_pos = prev_state["chef"]
        if chef_pos is None:
            return
        # Check proximity to pickable items in the state before the action
        for hold_idx, key in [
            (HOLD_CUTFISH, "cutFish"),
            (HOLD_CUTSHRIMP, "cutShrimp"),
            (HOLD_FISH, "fish"),
            (HOLD_SHRIMP, "shrimp"),
            (HOLD_PLATE, "plates"),
        ]:
            if self._nearest_item(chef_pos, prev_state[key]):
                self.held_item = hold_idx
                return

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, prev_state, curr_state, action):
        reward = -0.01  # time penalty

        prev_cut = len(prev_state["cutFish"]) + len(prev_state["cutShrimp"])
        curr_cut = len(curr_state["cutFish"]) + len(curr_state["cutShrimp"])
        if curr_cut > prev_cut:
            reward += 1.0

        # Serving: PICKUP/SETDOWN near the detected serving counter
        # while a processed ingredient disappears from the scene
        if action == ACTION_PICKUP and curr_state["chef"] is not None:
            serving_pos = curr_state["servingCounter"]
            if serving_pos is not None:
                chef_pos = curr_state["chef"]
                dist = (
                    (chef_pos[0] - serving_pos[0]) ** 2
                    + (chef_pos[1] - serving_pos[1]) ** 2
                ) ** 0.5
                if dist < 0.1 and curr_cut < prev_cut:
                    reward += 5.0

        return reward

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.held_item = HOLD_NONE
        
        # TODO: Need to somehow call the game to reset? or should this get called when the game is reset?

        frame = self._capture_frame()
        detections = self._detect_objects(frame)
        self.prev_state = self._build_game_state(detections)
        obs = self._encode_state(self.prev_state)

        return obs, {}

    def step(self, action):
        self.step_count += 1

        self._execute_action(action)

        frame = self._capture_frame()
        detections = self._detect_objects(frame)
        curr_state = self._build_game_state(detections)

        self._update_held_item(action, self.prev_state)
        obs = self._encode_state(curr_state)
        reward = self._compute_reward(self.prev_state, curr_state, action)
        self.prev_state = curr_state

        # where are we checking the time left for terminating
        terminated = False
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _execute_action(self, action):
        pyautogui.press(KEYMAP[action])

    def render(self):
        if self.render_mode == "human":
            frame = self._capture_frame()
            cv2.imshow("Overcooked RL Agent", frame)
            cv2.waitKey(1)

    def close(self):
        self.sct.close()
        cv2.destroyAllWindows()
