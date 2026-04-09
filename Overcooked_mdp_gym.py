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
}

# statoinary objects
STATIC_STATIONS = {
    "choppingBoards": [(396, 717), (505, 717), (1390, 717), (1502, 717)],
    "servingCounter": (1460, 373),
    "fishBox":        (414, 502),
    "shrimpBox":      (1500, 590),
}

# kitchen bounds
# [200:665, 400:1500]
KITCHEN_TOP = 200
KITCHEN_BOTTOM = 665
KITCHEN_LEFT = 400
KITCHEN_RIGHT = 1500

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

    def _normalize(self, px, py):
        nx = (px - self.monitor["left"]) / self.monitor["width"]
        ny = (py - self.monitor["top"])  / self.monitor["height"]
        return (nx, ny)

    def __init__(
        self,
        players: int = 1,
        render_mode: str = None,
        model_path: str = "objectTrackingModels/runs/detect/train2/weights/best.pt",
    ):
        super().__init__()
        self.players = players
        self.render_mode = render_mode

        # Set up the OpenCV screen and object detection model
        self.model = YOLO(model_path)
        self.sct = mss.mss()
         # TODO: MUST BE CHANGED BASED ON THE SCREEN CAPTURE
         # IF IT IS WHOLE SCREEN IT SHOULD BE 0, 0, (WIDTH OF SCREEN), (HEIGHT OF SCREEN)
        self.monitor = {"top": 200, "left": 0, "width": 600, "height": 400}

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CHOP', 'PICKUP']
        self.action_space = spaces.Discrete(len(self.actions))

        # Discrete state tuple (used as Q-table key):
        # (gx, gy)            chef tile position on 11x6 grid (-1,-1 if undetected)
        # held                held item: 0=none 1=plate 2=fish 3=shrimp 4=cutFish 5=cutShrimp
        # near_fishbox        bool: chef near fish supply box
        # near_shrimpbox      bool: chef near shrimp supply box
        # near_serving        bool: chef near serving counter
        # near_chopping_board bool: chef near chopping board
        # has_fish            bool: raw fish visible on counter
        # has_shrimp          bool: raw shrimp visible on counter
        # has_cut_fish        bool: cut fish visible on counter
        # has_cut_shrimp      bool: cut shrimp visible on counter
        # has_plate           bool: plate visible on counter
        # order               current order: 0=none 1=fish 2=shrimp
        self.observation_space = spaces.MultiDiscrete([11, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])

        self.max_steps = 500
        self.step_count = 0
        self.prev_state = None
        self.held_item = HOLD_NONE

        self.serving_counter_pos  = self._normalize(*STATIC_STATIONS["servingCounter"])
        self.fish_box_pos         = self._normalize(*STATIC_STATIONS["fishBox"])
        self.shrimp_box_pos       = self._normalize(*STATIC_STATIONS["shrimpBox"])
        self.chopping_board_pos   = [self._normalize(*p) for p in STATIC_STATIONS["choppingBoards"]]

        # Normalize kitchen bounds
        self.kitchen_left,  self.kitchen_top    = self._normalize(KITCHEN_LEFT,  KITCHEN_TOP)
        self.kitchen_right, self.kitchen_bottom = self._normalize(KITCHEN_RIGHT, KITCHEN_BOTTOM)

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
            "cutShrimp": []
        }
        class_to_key = {
            0: "chef",
            1: "plates",
            2: "fish",
            3: "shrimp",
            4: "cutFish",
            5: "cutShrimp"
        }
        for det in detections:
            key = class_to_key.get(det["class"])
            if key is None:
                continue
            if key == "chef":
                state[key] = det["center"]
            else:
                state[key].append(det["center"])
        return state

    def _encode_state(self, game_state, order):
        # chef location: 11x6 grid matching game tiles
        # if chef exists on the screen
        if game_state["chef"] is not None:
            cx, cy = game_state["chef"]
            # relational to kitchen bounds
            nx = (cx - self.kitchen_left) / (self.kitchen_right - self.kitchen_left)
            ny = (cy - self.kitchen_top)  / (self.kitchen_bottom - self.kitchen_top)
            # fits the location into a 11 x 6 grid to shrink grid sizes
            gx = int(np.clip(nx, 0, 0.999) * 11)
            gy = int(np.clip(ny, 0, 0.999) * 6)
        else:
            gx, gy = -1, -1

        # 2. Held item: 0-5
        held = self.held_item

        # 3. Proximity to key stations
        chef_pos = game_state["chef"]
        near_fishbox   = self._nearest_item(chef_pos, [self.fish_box_pos])
        near_shrimpbox = self._nearest_item(chef_pos, [self.shrimp_box_pos])
        near_serving   = self._nearest_item(chef_pos, [self.serving_counter_pos])
        near_chopping_board   = self._nearest_item(chef_pos, self.chopping_board_pos)

        # 4. What items exist on the counter
        has_fish       = len(game_state["fish"])      > 0
        has_shrimp     = len(game_state["shrimp"])    > 0
        has_cut_fish   = len(game_state["cutFish"])   > 0
        has_cut_shrimp = len(game_state["cutShrimp"]) > 0
        has_plate      = len(game_state["plates"])    > 0

        # 5. Current order from template matching: 0=none, 1=fish, 2=shrimp
        # comes from order passed in

        return (
            gx, gy,
            held,
            int(near_fishbox), int(near_shrimpbox), int(near_serving), int(near_chopping_board),
            int(has_fish), int(has_shrimp),
            int(has_cut_fish), int(has_cut_shrimp),
            int(has_plate),
            order,
        )
    

    # ------------------------------------------------------------------
    # Next order tracking
    # ------------------------------------------------------------------

    def _detect_order(self):
        # INPUT REAL COORDINATES
        order_region = {"top": 0, "left": 8, "width": 116, "height": 140}  # tune these coords
        screenshot = self.sct.grab(order_region)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2GRAY)

        best_label = 0  # 0 = no order detected
        best_score = 0.6

        for label, path in [(1, "templates/fish_order.png"), (2, "templates/shrimp_order.png")]:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_label = label

        return best_label  # 0=none, 1=fish, 2=shrimp

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
            serving_pos = self.serving_counter_pos
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
        order = self._detect_order()
        obs = self._encode_state(self.prev_state, order)

        return obs, {}

    def step(self, action):
        self.step_count += 1

        self._execute_action(action)

        frame = self._capture_frame()
        detections = self._detect_objects(frame)
        curr_state = self._build_game_state(detections)

        self._update_held_item(action, self.prev_state)
        order = self._detect_order()
        obs = self._encode_state(curr_state, order)
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
