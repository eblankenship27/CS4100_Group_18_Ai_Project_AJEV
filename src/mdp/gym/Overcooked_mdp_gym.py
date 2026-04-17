import time
import os

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
            model_path: str = "src/objectTrackingModels/runs/detect/train2/weights/best.pt",
            serving_pos: tuple = (1460, 373),
            fish_box_pos: tuple = (414, 502),
            shrimp_box_pos: tuple = (1500, 590),
    ):
        """
        serving_pos, fish_box_pos, shrimp_box_pos: optional (x, y) pixel coordinates
        within the monitor region. When provided, these override YOLO detection for
        those static objects. Coordinates are normalized internally.
        Example: serving_pos=(540, 200) means pixel (540, 200) inside the monitor window.
        """
        super().__init__()
        self.players = players
        self.render_mode = render_mode

        # Normalize manual pixel positions into [0,1] relative to the monitor region.
        # These are set before self.monitor is defined, so we store raw tuples and
        # normalize lazily in _build_game_state using self.monitor.
        self._manual_serving_pos = serving_pos
        self._manual_fish_box_pos = fish_box_pos
        self._manual_shrimp_box_pos = shrimp_box_pos

        # Set up the OpenCV screen and object detection model
        self.model = YOLO(model_path)
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # primary screen, full resolution

        w, h = self.monitor["width"], self.monitor["height"]

        self._static_positions = {
            "servingCounter": None,
            "fishBox": None,
            "shrimpBox": None,
        }

        if self._manual_serving_pos is not None:
            self._static_positions["servingCounter"] = (
                self._manual_serving_pos[0] / w,
                self._manual_serving_pos[1] / h
            )

        if self._manual_fish_box_pos is not None:
            self._static_positions["fishBox"] = (
                self._manual_fish_box_pos[0] / w,
                self._manual_fish_box_pos[1] / h
            )

        if self._manual_shrimp_box_pos is not None:
            self._static_positions["shrimpBox"] = (
                self._manual_shrimp_box_pos[0] / w,
                self._manual_shrimp_box_pos[1] / h
            )

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CHOP', 'PICKUP']
        self.action_space = spaces.Discrete(len(self.actions))

        # 31-float observation vector:
        # [0-1]   chef (x, y)
        # [2-7]   one-hot: held item (none/plate/fish/shrimp/cutFish/cutShrimp)
        # [8-9]   first plate center
        # [10-11] first fish center
        # [12-13] first shrimp center
        # [14-15] first cutFish center
        # [16-17] first cutShrimp center
        # [18-21] one-hot chef facing (up/down/left/right)
        # [22-23] one-hot current order (0=cutFish, 1=cutShrimp); -1 if undetected
        # [24-29] one-hot plate ingredient (same encoding as held item)
        # [30]    board reward given flag (1.0 = ingredient placed on board this order)
        #
        # Undetected objects default to (-1, -1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(31,), dtype=np.float32
        )
        
        self.serving_counter_pos  = self._normalize(*STATIC_STATIONS["servingCounter"])
        self.fish_box_pos         = self._normalize(*STATIC_STATIONS["fishBox"])
        self.shrimp_box_pos       = self._normalize(*STATIC_STATIONS["shrimpBox"])
        self.chopping_board_pos   = [self._normalize(*p) for p in STATIC_STATIONS["choppingBoards"]]

        # Normalize kitchen bounds
        self.kitchen_left,  self.kitchen_top    = self._normalize(KITCHEN_LEFT,  KITCHEN_TOP)
        self.kitchen_right, self.kitchen_bottom = self._normalize(KITCHEN_RIGHT, KITCHEN_BOTTOM)

        # TODO: Implement the chef facing values

        self.max_steps = 500
        self.step_count = 0
        self.prev_state = None
        self.held_item = HOLD_NONE
        self.plate_ingredient = HOLD_NONE
        self._board_reward_given = False
        self._plated_pickup_rewarded = False
        self._picked_correct_ingredient = False
        self.current_order = 0
        self.last_known_chef = (0.37512194315592445, 0.33323498478642216)  # chef spawn → grid (4, 2)

        # Statistics for chef timings and coordinates
        self.move_duration = 0.05
        self.chop_duration = 7.0

        self.chef_facing = ACTION_DOWN

        self.raw_static_pixels = {
            "fish_box": (414, 502),
            "shrimp_box": (1500, 590),
            "plate_dispenser": (1479, 500),
            "serving": (1460, 373),
            "cutting_boards": [
                (396, 717), (505, 717),
                (1390, 717), (1502, 717)
            ],
            "plates": [
                (650, 444), (748, 444),
                (1141, 444), (1240, 444)
            ]
        }

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

        for key in ["servingCounter", "fishBox", "shrimpBox"]:
            if self._static_positions[key] is not None:
                state[key] = self._static_positions[key]

        return state

    def _encode_state(self, game_state, order):
        obs = np.full(31, -1.0, dtype=np.float32)

        if game_state["chef"] is not None:
            self.last_known_chef = game_state["chef"]
            obs[0], obs[1] = game_state["chef"]
        elif self.last_known_chef is not None:
            obs[0], obs[1] = self.last_known_chef  # fallback to last known position

        # [2-7] held item one-hot
        obs[2 + self.held_item] = 1.0

        # [8-17] first detected instance of each dynamic object type
        for i, key in enumerate(["plates", "fish", "shrimp", "cutFish", "cutShrimp"]):
            if game_state[key]:
                obs[8 + 2 * i] = game_state[key][0][0]
                obs[8 + 2 * i + 1] = game_state[key][0][1]

        # [18-21] chef facing one-hot
        obs[18 + self.chef_facing] = 1.0

        # [22-23] order one-hot; _detect_order returns 0=none, 1=cutFish, 2=cutShrimp
        if order == 1:
            obs[22] = 1.0
        elif order == 2:
            obs[23] = 1.0
        elif order == 0:
            # if it can't find an order, default to fish order until it overrides
            obs[22] = 1.0

        # [24-29] plate ingredient one-hot (same encoding as held item)
        obs[24 + self.plate_ingredient] = 1.0

        # [30] board reward given flag — distinguishes episode-start from post-plating
        obs[30] = 1.0 if self._board_reward_given else 0.0

        return obs

    def _detect_order(self):
        order_region = {
            "top": 0, "left": 8, "width": 116, "height": 140
        }
        screenshot = self.sct.grab(order_region)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2GRAY)
        
        best_label = 0  # 0 = no order detected
        best_score = 0.6
        
        _tmpl_dir = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
        for label, path in [(1, os.path.join(_tmpl_dir, "fish_order.png")),
                            (2, os.path.join(_tmpl_dir, "shrimp_order.png"))]:
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

    def _serving_dist(self, chef_pos=None) -> float:
        """Euclidean distance (normalized) from chef to the serving counter."""
        if chef_pos is None:
            chef_pos = self.prev_state.get("chef") if self.prev_state else None
        if chef_pos is None:
            return 0.0
        sx, sy = self.serving_counter_pos
        return ((chef_pos[0] - sx) ** 2 + (chef_pos[1] - sy) ** 2) ** 0.5

    def _ingredient_source_dist(self, chef_pos=None) -> float:
        """Euclidean distance (normalized) from chef to the correct ingredient box."""
        if chef_pos is None:
            chef_pos = self.prev_state.get("chef") if self.prev_state else None
        if chef_pos is None or self.current_order == 0:
            return 0.0
        target = self.fish_box_pos if self.current_order == 1 else self.shrimp_box_pos
        return ((chef_pos[0] - target[0]) ** 2 + (chef_pos[1] - target[1]) ** 2) ** 0.5

    def _update_held_item(self, action, prev_state):
        if action != ACTION_PICKUP:
            return
        if self.held_item != HOLD_NONE:
            self.held_item = HOLD_NONE
            self.plate_ingredient = HOLD_NONE
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
                if hold_idx == HOLD_PLATE:
                    # Infer plate ingredient from nearby cut food (food sitting on the plate)
                    if self._nearest_item(chef_pos, prev_state["cutFish"]):
                        self.plate_ingredient = HOLD_CUTFISH
                    elif self._nearest_item(chef_pos, prev_state["cutShrimp"]):
                        self.plate_ingredient = HOLD_CUTSHRIMP
                    else:
                        self.plate_ingredient = HOLD_NONE
                return

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, prev_state, curr_state, action,
                        just_picked_ingredient=False, ingredient_correct=False,
                        just_picked_plated=False):
        reward = -0.01  # time penalty

        prev_cut = len(prev_state.get("cutFish", [])) + len(prev_state.get("cutShrimp", []))
        curr_cut = len(curr_state.get("cutFish", [])) + len(curr_state.get("cutShrimp", []))

        # Ingredient pickup
        if just_picked_ingredient:
            reward += 0.2 if ingredient_correct else -0.5

        # Chop detected — also acts as proxy for board placement since board placement
        # cannot be reliably detected from YOLO alone
        if curr_cut > prev_cut:
            if not self._board_reward_given and self._picked_correct_ingredient:
                reward += 2.0  # board placement bonus
            self._board_reward_given = True
            reward += 5.0 if self._picked_correct_ingredient else 0.0

        # Plated dish pickup (plate carrying a cut ingredient)
        if just_picked_plated and not self._plated_pickup_rewarded:
            correct = (
                (self.plate_ingredient == HOLD_CUTFISH   and self.current_order == 1) or
                (self.plate_ingredient == HOLD_CUTSHRIMP and self.current_order == 2)
            )
            if correct:
                reward += 5.0
                self._plated_pickup_rewarded = True

        # Serve: PICKUP near serving counter while a cut item disappears from the scene
        served = False
        if action == ACTION_PICKUP and curr_state.get("chef") is not None:
            chef_pos = curr_state["chef"]
            dist = (
                (chef_pos[0] - self.serving_counter_pos[0]) ** 2 +
                (chef_pos[1] - self.serving_counter_pos[1]) ** 2
            ) ** 0.5
            if dist < 0.1 and curr_cut < prev_cut:
                reward += 200.0
                served = True
                self._board_reward_given = False
                self._plated_pickup_rewarded = False
                self._picked_correct_ingredient = False

        return reward, served

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.held_item = HOLD_NONE
        self.plate_ingredient = HOLD_NONE
        self._board_reward_given = False
        self._plated_pickup_rewarded = False
        self._picked_correct_ingredient = False
        self.last_known_chef = (0.37512194315592445, 0.33323498478642216)

        # TODO: Need to somehow call the game to reset? or should this get called when the game is reset?

        frame = self._capture_frame()
        detections = self._detect_objects(frame)
        self.prev_state = self._build_game_state(detections)
        order = self._detect_order()
        self.current_order = order
        obs = self._encode_state(self.prev_state, order)

        return obs, {}

    def step(self, action):
        self.step_count += 1

        # Capture pre-action state for potential-based shaping
        prev_held = self.held_item
        carrying_plated = (
            self.held_item == HOLD_PLATE and
            self.plate_ingredient in (HOLD_CUTFISH, HOLD_CUTSHRIMP)
        )
        pre_chef_pos = self.prev_state.get("chef") if self.prev_state else None
        pre_serve_dist = self._serving_dist(pre_chef_pos) if carrying_plated else None
        pre_src_dist   = (
            self._ingredient_source_dist(pre_chef_pos)
            if (self.held_item == HOLD_NONE and not self._board_reward_given)
            else None
        )

        self._execute_action(action)

        frame = self._capture_frame()
        detections = self._detect_objects(frame)
        curr_state = self._build_game_state(detections)

        # Detect order before updating held item so correctness can be checked
        order = self._detect_order()
        self.current_order = order

        self._update_held_item(action, self.prev_state)

        # Detect ingredient pickup event by comparing held item before and after
        just_picked_ingredient = (prev_held == HOLD_NONE and
                                  self.held_item in (HOLD_FISH, HOLD_SHRIMP))
        if just_picked_ingredient:
            ingredient_correct = (
                (self.held_item == HOLD_FISH   and order == 1) or
                (self.held_item == HOLD_SHRIMP and order == 2)
            )
            self._picked_correct_ingredient = ingredient_correct
        else:
            ingredient_correct = False

        # Detect plated dish pickup (empty-handed → holding plate with ingredient)
        just_picked_plated = (
            prev_held == HOLD_NONE and
            self.held_item == HOLD_PLATE and
            self.plate_ingredient in (HOLD_CUTFISH, HOLD_CUTSHRIMP)
        )

        obs = self._encode_state(curr_state, order)
        reward, served = self._compute_reward(
            self.prev_state, curr_state, action,
            just_picked_ingredient=just_picked_ingredient,
            ingredient_correct=ingredient_correct,
            just_picked_plated=just_picked_plated,
        )

        curr_chef_pos = curr_state.get("chef")

        # Serving shaping: reward steps toward serving counter while carrying plated dish
        if pre_serve_dist is not None and not served:
            if (self.held_item == HOLD_PLATE and
                    self.plate_ingredient in (HOLD_CUTFISH, HOLD_CUTSHRIMP)):
                reward += (pre_serve_dist - self._serving_dist(curr_chef_pos)) * 0.15

        # Ingredient source shaping: reward steps toward correct box while empty and pre-board
        if (pre_src_dist is not None and
                not just_picked_ingredient and
                not self._board_reward_given and
                self.held_item == HOLD_NONE):
            reward += (pre_src_dist - self._ingredient_source_dist(curr_chef_pos)) * 0.1

        self.prev_state = curr_state

        terminated = False
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _execute_action(self, action):
        # TODO: Implement actions to correlate from discretized state to the actual game

        # Examples:
        # - Set the movement to hold and do it
        # - When pickup/putdown, move towards tile slightly?

        key = KEYMAP[action]

        if action in [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]:
            self.chef_facing = action

            pyautogui.keyDown(key)
            time.sleep(self.move_duration)
            pyautogui.keyUp(key)

        elif action == ACTION_CHOP:
            pyautogui.press(key)
            time.sleep(self.chop_duration)

        elif action == ACTION_PICKUP:
            pyautogui.press(key)
            time.sleep(0.1)

    def render(self):
        if self.render_mode == "human":
            frame = self._capture_frame()
            cv2.imshow("Overcooked RL Agent", frame)
            cv2.waitKey(1)

    def close(self):
        self.sct.close()
        cv2.destroyAllWindows()
