import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Tile types
TILE_EMPTY=0
TILE_WALL=1
TILE_FISH_BOX=2
TILE_SHRIMP_BOX=3
TILE_CUTTING_BOARD=4
TILE_PLATE_SHELF=5
TILE_SERVING=6

# Held item indices - same as real gym
HOLD_NONE=0
HOLD_PLATE=1
HOLD_FISH=2
HOLD_SHRIMP=3
HOLD_CUTFISH=4
HOLD_CUTSHRIMP=5

# Action Contants - same as real gym
ACTION_UP=0
ACTION_DOWN=1
ACTION_LEFT=2
ACTION_RIGHT=3
ACTION_CHOP=4
ACTION_PICKUP=5

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
    TILE_EMPTY:         '.',
    TILE_WALL:          '#',
    TILE_FISH_BOX:      'F',
    TILE_SHRIMP_BOX:    'S',
    TILE_CUTTING_BOARD: 'C',
    TILE_PLATE_SHELF:   'P',
    TILE_SERVING:       'V',
}
_ITEM_SYMBOLS = {
    "fish":       'f',
    "shrimp":     's',
    "cutFish":    'q',
    "cutShrimp":  'r',
    "plate":      'p',
}
_HOLD_NAMES = {
    HOLD_NONE:      'nothing',
    HOLD_PLATE:     'plate',
    HOLD_FISH:      'fish',
    HOLD_SHRIMP:    'shrimp',
    HOLD_CUTFISH:   'cutFish',
    HOLD_CUTSHRIMP: 'cutShrimp',
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
        self._spawn_cells = [(3,2),(9,2)]
        
        for r, row in enumerate(self.layout):
            for c, tile in enumerate(row):
                nx = c / (self.GRID_COLS - 1)
                ny = r / (self.GRID_ROWS - 1)
                if tile == TILE_SERVING:
                    self._serving_cells.append((r, c))
                elif tile == TILE_FISH_BOX and "fish_box" not in self._static_positions:
                    self._static_positions["fish_box"] = (nx, ny)
                elif tile == TILE_SHRIMP_BOX and "shrimp_box" not in self._static_positions:
                    self._static_positions["shrimp_box"] = (nx, ny)
                elif tile == TILE_CUTTING_BOARD:
                    self._cutting_board_cells.append((r, c))
                elif tile == TILE_PLATE_SHELF:
                    self._plate_shelf_cell.append((r, c))
                    
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        
        self.chef_row = self._spawn_cells[0][0]
        self.chef_col = self._spawn_cells[0][1]
        self.held_item = HOLD_NONE
        self.plate_had_ingredient = False
        self.world_items = [
            {"type": "plate", "row": 4, "col": 4},
            {"type": "plate", "row": 4, "col": 3},
            {"type": "plate", "row": 4, "col": 8},
            {"type": "plate", "row": 4, "col": 9},
        ] # list of {"type": str, "row": int, "col": int}
        self.order = None
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.chef_row, self.chef.col = self._spawn_cell
        self.held_item = HOLD_NONE
        self.plate_has_ingredient = False
        
        self.order = np.random.choice(["cutFish", "cutShrimp"])
        
        if self.render_mode == "human":
            self.render()
            
        return self._encode_obs(), {}
    
    def step(self, action: int):
        self.step_count += 1
        
        events = {}
        if action == ACTION_UP:
            events = self._try_move(-1, 0)
        elif action == ACTION_DOWN:
            events = self._try_move(1, 0)
        elif action == ACTION_LEFT:
            events = self._try_move(0, -1)
        elif action == ACTION_RIGHT:
            events = self._try_move(0, 1)
        elif action == ACTION_CHOP:
            events = self._try_chop()
        elif action == ACTION_PICKUP:
            events = self._try_pickup_or_serve()
            
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
        
        print(f"\nStep {self.step_count}  holding={_HOLD_NAMES[self.held_item]}  order={self.pending_order}")
        for row in grid:
            print(' '.join(row))
            
    def close(self):
        pass

    def _try_move(self, dr: int, dc: int) -> dict:
        new_r = self.chef_row + dr
        new_c = self.chef_col + dc
        if (0 <= new_r < self.GRID_ROWS) and 0 <= new_c < self.GRID_COLS and self.layout[new_r][new_c]:
            self.chef_row = new_r
            self.chef_col = new_c
        return {}
    
    def _try_chop(self) -> dict:
        # Define the action of the agent trying to chop an item at a cutting board
        pass
        
    def _try_pickup_or_putdown(self) -> dict:
        # Define the action of the agent trying to pick up and item, or put it down
        pass

    def _adjacent_cells(self, row: int, col: int):
        # Return the cells that are adjacent to the player location
        pass
        
    def _encode_obs(self) -> np.ndarray:
        # Encode the current observation state into an array for state handling
        pass

    def _compute_reward(self, events: dict) -> float:
        # Compute the rewards that the agent shuold receive based on the events that occured
        pass
        