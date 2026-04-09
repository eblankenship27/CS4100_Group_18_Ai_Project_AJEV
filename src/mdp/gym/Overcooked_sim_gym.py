import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Tile types
TILE_EMPTY = 0
TILE_WALL = 1
TILE_FISH_BOX = 2
TILE_SHRIMP_BOX = 3
TILE_CUTTING_BOARD = 4
TILE_PLATE_SHELF = 5
TILE_SERVING = 6

# Held item indices - same as real gym
HOLD_NONE = 0
HOLD_PLATE = 1
HOLD_FISH = 2
HOLD_SHRIMP = 3
HOLD_CUTFISH = 4
HOLD_CUTSHRIMP = 5

# Action Contants - same as real gym
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CHOP = 4
ACTION_PICKUP = 5

LEVEL_ONE_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
    [2, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 5],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# ASCII symbols for render()
_TILE_SYMBOLS = {
    TILE_EMPTY: '.',
    TILE_WALL: '#',
    TILE_FISH_BOX: 'F',
    TILE_SHRIMP_BOX: 'S',
    TILE_CUTTING_BOARD: 'C',
    TILE_PLATE_SHELF: 'P',
    TILE_SERVING: 'V',
}
_ITEM_SYMBOLS = {
    "fish": 'f',
    "shrimp": 's',
    "cutFish": 'q',
    "cutShrimp": 'r',
    "plate": 'p',
}
_HOLD_NAMES = {
    HOLD_NONE: 'nothing',
    HOLD_PLATE: 'plate',
    HOLD_FISH: 'fish',
    HOLD_SHRIMP: 'shrimp',
    HOLD_CUTFISH: 'cutFish',
    HOLD_CUTSHRIMP: 'cutShrimp',
}
_NAME_TO_HOLD = {v: k for k, v in _HOLD_NAMES.items()}
_DIR_TO_FACING = {
    (-1, 0): ACTION_UP,
    (1, 0): ACTION_DOWN,
    (0, -1): ACTION_LEFT,
    (0, 1): ACTION_RIGHT,
}
_FACING_TO_DIR = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class OvercookedSimEnv(gym.Env):

    def __init__(self, layout=None, spawn=[], max_steps=500, render_mode=None):

        super().__init__()

        self.layout = layout if layout is not None else LEVEL_ONE_LAYOUT
        self.GRID_ROWS = len(self.layout)
        self.GRID_COLS = len(self.layout[0])
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Scan layout to populate special cells
        self._static_positions = {}
        self._serving_cells = []
        self._cutting_board_cells = []
        self._plate_shelf_cell = None
        self._spawn_cells = [(3, 2), (9, 2)]

        for r, row in enumerate(self.layout):
            for c, tile in enumerate(row):
                nx = c / (self.GRID_COLS - 1)
                ny = r / (self.GRID_ROWS - 1)
                if tile == TILE_SERVING:
                    self._serving_cells.append((r, c))
                    if "serving" not in self._static_positions:
                        self._static_positions["serving"] = (nx, ny)
                elif tile == TILE_FISH_BOX and "fish_box" not in self._static_positions:
                    self._static_positions["fish_box"] = (nx, ny)
                elif tile == TILE_SHRIMP_BOX and "shrimp_box" not in self._static_positions:
                    self._static_positions["shrimp_box"] = (nx, ny)
                elif tile == TILE_CUTTING_BOARD:
                    self._cutting_board_cells.append((r, c))
                elif tile == TILE_PLATE_SHELF and "plate_shelf" not in self._static_positions:
                    self._static_positions["plate_shelf"] = (nx, ny)
                    self._plate_shelf_cell = (r, c)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.chef_row = self._spawn_cells[0][0]
        self.chef_col = self._spawn_cells[0][1]
        self.chef_facing = ACTION_DOWN
        self.held_item = HOLD_NONE
        self.plate_ingredient = HOLD_NONE
        self.world_items = [
            {"type": "plate", "row": 4, "col": 4},
            {"type": "plate", "row": 4, "col": 3},
            {"type": "plate", "row": 4, "col": 8},
            {"type": "plate", "row": 4, "col": 9},
        ]  # list of {"type": str, "row": int, "col": int}
        self.order = np.random.choice([
            'cutFish', 'cutShrimp'
        ])
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.chef_row, self.chef_col = self._spawn_cells[0]
        self.chef_facing = ACTION_DOWN
        self.held_item = HOLD_NONE
        self.plate_has_ingredient = HOLD_NONE

        self.order = np.random.choice(["cutFish", "cutShrimp"])

        if self.render_mode == "human":
            self.render()

        return self._encode_obs(), {}

    def step(self, action: int):
        self.step_count += 1

        # TODO: implement plate shelf to continously replace the plates

        events = {}

        self._refill_plates()

        if action == ACTION_CHOP:
            events = self._try_chop()
        elif action == ACTION_PICKUP:
            events = self._try_pickup_or_putdown()
        else:
            events = self._try_move(_FACING_TO_DIR[action])

        reward = self._compute_reward(events)
        terminated = False
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._encode_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return

        grid = [
            [_TILE_SYMBOLS.get(self.layout[r][c], '?') for c in range(self.GRID_COLS)]
            for r in range(self.GRID_ROWS)
        ]

        for item in self.world_items:
            grid[item["row"]][item["col"]] = _ITEM_SYMBOLS.get(item["type"], '?')

        grid[self.chef_row][self.chef_col] = '@'

        print(f"\nStep {self.step_count}  holding={_HOLD_NAMES[self.held_item]}  order={self.order}")
        for row in grid:
            print(' '.join(row))

    def close(self):
        pass

    def _try_move(self, dr: int, dc: int) -> dict:
        new_r = self.chef_row + dr
        new_c = self.chef_col + dc
        self.chef_facing = _DIR_TO_FACING[(dr, dc)]

        if (0 <= new_r < self.GRID_ROWS) and 0 <= new_c < self.GRID_COLS and self.layout[new_r][new_c] == TILE_EMPTY:
            self.chef_row = new_r
            self.chef_col = new_c
        return {}

    def _try_chop(self) -> dict:
        # Define the action of the agent trying to chop an item at a cutting board

        # TODO: Implement plate shelf where you can't place things on it?

        cell = self._facing_cell()
        if cell is None:
            return {}
        fr, fc = cell
        for item in self.world_items:
            if item['type'] in ('fish', 'shrimp') and item['row'] == fr and item['col'] == fc:
                for br, bc in self._cutting_board_cells:
                    if br == fr and bc == fc:
                        self.world_items.remove(item)
                        self.world_items.append({
                            'type': _HOLD_NAMES[HOLD_CUTFISH if item['type'] == 'fish' else HOLD_CUTSHRIMP],
                            'row': item['row'],
                            'col': item['col'],
                        })
                        return {"chopped": True}
        return {}

    def _try_pickup_or_putdown(self) -> dict:

        # Define the action of the agent trying to pick up and item, or put it down

        # Get the cell the agent is facing
        cell = self._facing_cell()
        if cell is None:
            return {}
        r, c = cell

        if self.layout[r][c] == TILE_PLATE_SHELF:
            return {}

        if self.held_item == HOLD_PLATE and self.plate_ingredient in (HOLD_CUTFISH, HOLD_CUTSHRIMP):
            if self.layout[r][c] == TILE_SERVING:
                # If served with a plate and a cut food on the plate then should serve and remove the item
                self.held_item = HOLD_NONE
                self.plate_ingredient = HOLD_NONE
                self.order = np.random.choice(['cutFish', 'cutShrimp'])
                return {"served": True}

            for item in self.world_items:
                if item['row'] == r and item['col'] == c:
                    return {}

            self.world_items.append({
                'type': _HOLD_NAMES[self.held_item],
                'row': r,
                'col': c,
                'holds': _HOLD_NAMES[self.plate_ingredient]
            })
            self.held_item = HOLD_NONE
            self.plate_ingredient = HOLD_NONE
            return {}

        if self.held_item in (HOLD_CUTFISH, HOLD_CUTSHRIMP):
            if self.layout[r][c] == TILE_SERVING:
                # If served without a plate, should remove the item but not serve
                self.held_item = HOLD_NONE
                return {'bad_serve': True}

            # Check each item
            for item in self.world_items:
                # If there is a item at the facing cell
                if item['row'] == r and item['col'] == c:
                    # If is a plate, put on the plate
                    if item['type'] == 'plate' and item['holds'] == 'nothing':
                        self.world_items.remove(item)
                        self.world_items.append({
                            'type': 'plate',
                            'row': r,
                            'col': c,
                            'holds': _HOLD_NAMES[self.held_item]
                        })
                        self.held_item = HOLD_NONE
                    # Else return, as you can't put it down
                    return {}
            self.world_items.append({
                'type': _HOLD_NAMES[self.held_item],
                'row': r,
                'col': c,
                'holds': _HOLD_NAMES[HOLD_NONE]
            })
            self.held_item = HOLD_NONE
            return {}

        if self.held_item in (HOLD_FISH, HOLD_SHRIMP):
            # If holding a unprepped food and put on plate, do nothing
            # If holding a unprepped food and put on serve, get rid of food but not complete order
            if self.layout[r][c] == TILE_SERVING:
                # If served unprepped, should remove the item but not serve
                self.held_item = HOLD_NONE
                return {'bad_serve': True}

            # If holding unprepped food and try to place on another item, do nothing
            for item in self.world_items:
                if item['row'] == r and item['col'] == c:
                    return {}

            # If no items in facing cell, place held item
            self.world_items.append({
                'type': _HOLD_NAMES[self.held_item],
                'row': r,
                'col': c,
                'holds': _HOLD_NAMES[HOLD_NONE]
            })
            self.held_item = HOLD_NONE
            return {}
        # If hold plate and pick up food, the plate gets put down on the food,

        if self.held_item == HOLD_PLATE:

            if self.layout[r][c] == TILE_SERVING:
                # If served just plate, should remove the item but not serve
                self.held_item = HOLD_NONE
                return {'bad_serve': True}

            for item in self.world_items:
                if item['row'] == r and item['col'] == c:
                    if item['type'] in (_HOLD_NAMES[HOLD_CUTFISH], _HOLD_NAMES[HOLD_CUTSHRIMP]):
                        # If placing plate on prepped food, them create place with prepped food
                        self.world_items.remove(item)
                        self.world_items.append({
                            'type': 'plate',
                            'row': r,
                            'col': c,
                            'holds': _HOLD_NAMES[item['type']]
                        })
                        self.held_item = HOLD_NONE

                    return {}

            self.world_items.append({
                'type': _HOLD_NAMES[self.held_item],
                'row': r,
                'col': c,
                'holds': _HOLD_NAMES[HOLD_NONE]
            })
            self.held_item = HOLD_NONE
            return {}

        # If hold plate and try to put on unprepped food, do nothing

        # If hold unprepped food and try to put on plate, do nothing
        if self.held_item == HOLD_NONE:

            if self.layout[r][c] in (TILE_FISH_BOX, TILE_SHRIMP_BOX):
                self.held_item = HOLD_FISH if self.layout[r][c] == TILE_FISH_BOX else HOLD_SHRIMP
                return {'ingredient': True}

            for item in self.world_items:
                if item['row'] == r and item['col'] == c:
                    self.held_item = _NAME_TO_HOLD[item['type']]
                    self.world_items.remove(item)
                    return {}

            return {}

        return {}

    # HELPER FUNCTIONS

    def _adjacent_cells(self, row: int, col: int, v_facing: int, h_facing: int):
        # Return the cells that are adjacent to the player location
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                neighbors.append((nr, nc))
        return neighbors

    def _facing_cell(self):
        dr, dc = _FACING_TO_DIR[self.chef_facing]
        r, c = self.chef_row + dr, self.chef_col + dc
        if 0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS:
            return (r, c)

        return None

    def _compute_reward(self, events: dict) -> float:
        # Compute the rewards that the agent shuold receive based on the events that occured
        reward = -0.01
        if events.get("chopped"):
            reward += 1.0
        if events.get("served"):
            reward += 20.0
        if events.get("bad_serve"):
            reward += -3.0
        return reward

    # Observation Encoding

    def _encode_obs(self) -> np.ndarray:
        # Encode the current observation state into an array for state handling
        obs = np.full(28, -1.0, dtype=np.float32)

        # [0-1] Chef Position
        obs[0] = self.chef_col / (self.GRID_COLS - 1)
        obs[1] = self.chef_row / (self.GRID_ROWS - 1)

        # [2-7] One hot held item
        obs[2 + self.held_item] = 1.0

        # [8-17] First World item of each dynamic type
        for i, itype in enumerate(["plate", "fish", "shrimp", "cutFish", "cutShrimp"]):
            for item in self.world_items:
                if item["type"] == itype:
                    obs[8 + (2 * i)] = item["col"] / (self.GRID_COLS - 1)
                    obs[8 + (2 * i) + 1] = item["row"] / (self.GRID_ROWS - 1)
                    break

        # [18-19] Serving counter (static)
        if "serving" in self._static_positions:
            obs[18], obs[19] = self._static_positions["serving"]

        # [20-21] Fish box (static)
        if "fish_box" in self._static_positions:
            obs[20], obs[21] = self._static_positions["fish_box"]

        # [22-23] Shrimp box (static)
        if "shrimp_box" in self._static_positions:
            obs[22], obs[23] = self._static_positions["shrimp_box"]

        # [24-27] Player facing
        obs[24 + self.chef_facing] = 1.0

        return obs

    def _refill_plates(self):
        for r, c in [self._plate_shelf_cell]:
            plate_exists = any(
                item["row"] == r and item["col"] == c and item["type"] == "plate"
                for item in self.world_items
            )
            if not plate_exists:
                self.world_items.append({
                    "type": "plate",
                    "row": r,
                    "col": c,
                    "holds": "nothing"
                })
