import pygame
import sys
import time
from Overcooked_sim_gym import (
    OvercookedSimEnv,
    TILE_EMPTY, TILE_WALL, TILE_FISH_BOX, TILE_SHRIMP_BOX,
    TILE_CUTTING_BOARD, TILE_PLATE_SHELF, TILE_SERVING,
    HOLD_NONE, HOLD_PLATE, HOLD_FISH, HOLD_SHRIMP, HOLD_CUTFISH, HOLD_CUTSHRIMP,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_CHOP, ACTION_PICKUP,
    _HOLD_NAMES, _FACING_TO_DIR,
)

# ── Display constants ──────────────────────────────────────────────────────────
CELL_SIZE     = 68
CONSOLE_WIDTH = 370

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0  )
GRAY       = (210, 210, 210)
DARK_GRAY  = (55,  55,  55 )
BLUE       = (30,  80,  200)

TILE_COLORS = {
    TILE_EMPTY:         (245, 240, 220),
    TILE_WALL:          (55,  55,  55 ),
    TILE_FISH_BOX:      (65, 115, 210),
    TILE_SHRIMP_BOX:    (215, 75,  145),
    TILE_CUTTING_BOARD: (145, 88,  28 ),
    TILE_PLATE_SHELF:   (175, 198, 220),
    TILE_SERVING:       (235, 185, 20 ),
}

TILE_LABEL_COLOR = {
    TILE_EMPTY:         (190, 180, 160),
    TILE_WALL:          (110, 110, 110),
    TILE_FISH_BOX:      WHITE,
    TILE_SHRIMP_BOX:    WHITE,
    TILE_CUTTING_BOARD: WHITE,
    TILE_PLATE_SHELF:   DARK_GRAY,
    TILE_SERVING:       DARK_GRAY,
}

TILE_LABELS = {
    TILE_WALL:          '#',
    TILE_FISH_BOX:      'FISH\nBOX',
    TILE_SHRIMP_BOX:    'SHR\nBOX',
    TILE_CUTTING_BOARD: 'CUT',
    TILE_PLATE_SHELF:   'PLT\nSHLF',
    TILE_SERVING:       'SERVE',
}

ITEM_COLORS = {
    'plate':     (252, 252, 252),
    'fish':      (65,  115, 210),
    'shrimp':    (215, 75,  145),
    'cutFish':   (130, 178, 240),
    'cutShrimp': (240, 138, 185),
}

HOLD_BADGE_COLOR = {
    HOLD_PLATE:     (252, 252, 252),
    HOLD_FISH:      (65,  115, 210),
    HOLD_SHRIMP:    (215, 75,  145),
    HOLD_CUTFISH:   (130, 178, 240),
    HOLD_CUTSHRIMP: (240, 138, 185),
}

ACTION_NAMES = {
    ACTION_UP:     'Up',
    ACTION_DOWN:   'Down',
    ACTION_LEFT:   'Left',
    ACTION_RIGHT:  'Right',
    ACTION_CHOP:   'Chop',
    ACTION_PICKUP: 'Pickup',
}

FACING_NAMES = {
    ACTION_UP:    'Up',
    ACTION_DOWN:  'Down',
    ACTION_LEFT:  'Left',
    ACTION_RIGHT: 'Right',
}

# ── Module-level state ────────────────────────────────────────────────────────
screen     = None
clock      = None
game       = None
game_ended = False
action_log = []          # rolling list[str] of recent step results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cell_rect(row: int, col: int) -> pygame.Rect:
    return pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)


def _blit_centred(surface, text: str, font, colour, cx: int, cy: int):
    surf = font.render(text, True, colour)
    surface.blit(surf, surf.get_rect(center=(cx, cy)))


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_tile(row: int, col: int, tile: int):
    rect = _cell_rect(row, col)
    pygame.draw.rect(screen, TILE_COLORS.get(tile, GRAY), rect)
    pygame.draw.rect(screen, (105, 105, 105), rect, 1)

    label = TILE_LABELS.get(tile)
    if label is None:
        return
    fc   = TILE_LABEL_COLOR.get(tile, BLACK)
    font = pygame.font.Font(None, 17)
    lines = label.split('\n')
    total_h = len(lines) * 17
    y0 = rect.centery - total_h // 2 + 8
    for i, line in enumerate(lines):
        _blit_centred(screen, line, font, fc, rect.centerx, y0 + i * 17)


def _draw_item(item: dict):
    row, col = item['row'], item['col']
    rect = _cell_rect(row, col)
    cx, cy = rect.centerx, rect.centery
    itype = item['type']
    holds = item.get('holds', 'nothing')

    if itype == 'plate':
        r = CELL_SIZE // 3
        pygame.draw.circle(screen, ITEM_COLORS['plate'], (cx, cy), r)
        pygame.draw.circle(screen, (155, 155, 155), (cx, cy), r, 2)
        pygame.draw.circle(screen, (220, 220, 220), (cx, cy), r - 5, 1)
        if holds != 'nothing':
            ing_col = ITEM_COLORS.get(holds, GRAY)
            pygame.draw.circle(screen, ing_col, (cx, cy), r // 2)
            short = {'fish': 'F', 'shrimp': 'S', 'cutFish': 'cF', 'cutShrimp': 'cS'}.get(holds, '?')
            _blit_centred(screen, short, pygame.font.Font(None, 15), BLACK, cx, cy)
    else:
        fill = ITEM_COLORS.get(itype, GRAY)
        hw, hh = CELL_SIZE // 3, CELL_SIZE // 4
        ell_rect = pygame.Rect(cx - hw, cy - hh, hw * 2, hh * 2)
        pygame.draw.ellipse(screen, fill, ell_rect)
        pygame.draw.ellipse(screen, BLACK, ell_rect, 1)
        short = {'fish': 'fish', 'shrimp': 'shmp', 'cutFish': 'cFish', 'cutShrimp': 'cShmp'}.get(itype, itype)
        _blit_centred(screen, short, pygame.font.Font(None, 15), BLACK, cx, cy)
        if itype in ('cutFish', 'cutShrimp'):
            pygame.draw.line(screen, (190, 45, 45), (cx - 10, cy + 8), (cx + 10, cy + 8), 2)


def _draw_held_badge(cx: int, cy: int):
    """Small badge on the chef showing what is held (and plate ingredient)."""
    if game.held_item == HOLD_NONE:
        return
    col   = HOLD_BADGE_COLOR.get(game.held_item, GRAY)
    bx    = cx + CELL_SIZE // 4
    by    = cy - CELL_SIZE // 4
    r     = 9
    pygame.draw.circle(screen, col, (bx, by), r)
    pygame.draw.circle(screen, BLACK, (bx, by), r, 1)
    short = {HOLD_PLATE: 'P', HOLD_FISH: 'F', HOLD_SHRIMP: 'S',
             HOLD_CUTFISH: 'cF', HOLD_CUTSHRIMP: 'cS'}.get(game.held_item, '?')
    _blit_centred(screen, short, pygame.font.Font(None, 14), BLACK, bx, by)

    if game.held_item == HOLD_PLATE and game.plate_ingredient != HOLD_NONE:
        ing_col = HOLD_BADGE_COLOR.get(game.plate_ingredient, GRAY)
        pygame.draw.circle(screen, ing_col, (bx + 10, by - 5), 5)
        pygame.draw.circle(screen, BLACK, (bx + 10, by - 5), 5, 1)


def _draw_chef():
    rect = _cell_rect(game.chef_row, game.chef_col)
    cx, cy = rect.centerx, rect.centery
    TEAL       = (0,  155, 155)
    TEAL_DARK  = (0,  100, 100)

    # Drop shadow
    pygame.draw.circle(screen, TEAL_DARK, (cx + 2, cy + 2), CELL_SIZE // 3)
    # Body
    pygame.draw.circle(screen, TEAL, (cx, cy), CELL_SIZE // 3)
    pygame.draw.circle(screen, TEAL_DARK, (cx, cy), CELL_SIZE // 3, 2)

    # Facing arrow
    dr, dc = _FACING_TO_DIR[game.chef_facing]
    alen   = CELL_SIZE // 3 - 4
    ax, ay = cx + dc * alen, cy + dr * alen
    pygame.draw.line(screen, WHITE, (cx, cy), (ax, ay), 3)
    pygame.draw.circle(screen, WHITE, (ax, ay), 4)

    # Chef label
    _blit_centred(screen, '@', pygame.font.Font(None, 22), WHITE, cx, cy)

    # Held-item badge
    _draw_held_badge(cx, cy)


def _draw_console():
    grid_w = game.GRID_COLS * CELL_SIZE
    grid_h = game.GRID_ROWS * CELL_SIZE
    cx0    = grid_w

    # Panel
    pygame.draw.rect(screen, GRAY, pygame.Rect(cx0, 0, CONSOLE_WIDTH, grid_h))
    pygame.draw.line(screen, DARK_GRAY, (cx0, 0), (cx0, grid_h), 2)

    font_title = pygame.font.Font(None, 30)
    font_info  = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 19)

    x = cx0 + 12
    y = 10

    # Title
    screen.blit(font_title.render("Overcooked!", True, BLUE), (x, y))
    y += 38

    # ── Game state ────────────────────────────────────────────────────────────
    held_str   = _HOLD_NAMES.get(game.held_item, '?')
    plate_str  = _HOLD_NAMES.get(game.plate_ingredient, '?') if game.plate_ingredient != HOLD_NONE else 'empty'
    order_col  = (0, 150, 50) if game.order == 'cutFish' else (175, 0, 115)
    facing_str = FACING_NAMES.get(game.chef_facing, '?')

    info_lines = [
        (f"Steps   : {game.step_count} / {game.max_steps}", BLACK),
        (f"Order   : {game.order}",                         order_col),
        (f"Holding : {held_str}",                           DARK_GRAY),
        (f"Plate   : {plate_str}",                          DARK_GRAY),
        (f"Pos     : row={game.chef_row}  col={game.chef_col}", DARK_GRAY),
        (f"Facing  : {facing_str}",                         DARK_GRAY),
    ]
    for text, colour in info_lines:
        screen.blit(font_info.render(text, True, colour), (x, y))
        y += 26

    # Steps progress bar
    bar_w   = CONSOLE_WIDTH - 24
    bar_h   = 7
    bar_x   = x
    bar_y   = y + 2
    ratio   = min(game.step_count / max(game.max_steps, 1), 1.0)
    bar_col = (
        (60, 180, 60) if ratio < 0.6 else
        (220, 160, 0) if ratio < 0.85 else
        (200, 50, 50)
    )
    pygame.draw.rect(screen, (170, 170, 170), pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=3)
    pygame.draw.rect(screen, bar_col,         pygame.Rect(bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=3)
    y += 18

    y += 4
    pygame.draw.line(screen, DARK_GRAY, (x, y), (x + CONSOLE_WIDTH - 24, y), 1)
    y += 10

    # ── Colour legend ─────────────────────────────────────────────────────────
    screen.blit(font_info.render("Legend:", True, BLUE), (x, y))
    y += 26
    legend = [
        (ITEM_COLORS['fish'],      "Fish / Cut Fish"),
        (ITEM_COLORS['shrimp'],    "Shrimp / Cut Shrimp"),
        (ITEM_COLORS['plate'],     "Plate"),
        ((0, 155, 155),            "Chef  (@)"),
        (TILE_COLORS[TILE_SERVING],"Serving counter"),
        (TILE_COLORS[TILE_CUTTING_BOARD], "Cutting board"),
    ]
    for swatch, label in legend:
        pygame.draw.rect(screen, swatch,  pygame.Rect(x, y + 2, 14, 14))
        pygame.draw.rect(screen, DARK_GRAY, pygame.Rect(x, y + 2, 14, 14), 1)
        screen.blit(font_small.render(label, True, BLACK), (x + 20, y))
        y += 20

    y += 6
    pygame.draw.line(screen, DARK_GRAY, (x, y), (x + CONSOLE_WIDTH - 24, y), 1)
    y += 10

    # ── Controls ──────────────────────────────────────────────────────────────
    screen.blit(font_info.render("Controls:", True, BLUE), (x, y))
    y += 24
    controls = [
        "W / \u2191      Move Up",
        "S / \u2193      Move Down",
        "A / \u2190      Move Left",
        "D / \u2192      Move Right",
        "C            Chop",
        "E / Space    Pickup / Put-down",
        "R            Reset",
    ]
    for ctrl in controls:
        screen.blit(font_small.render(ctrl, True, DARK_GRAY), (x, y))
        y += 20

    y += 6
    pygame.draw.line(screen, DARK_GRAY, (x, y), (x + CONSOLE_WIDTH - 24, y), 1)
    y += 10

    # ── Action log ────────────────────────────────────────────────────────────
    screen.blit(font_info.render("Recent Actions:", True, BLUE), (x, y))
    y += 26
    max_entries = (grid_h - y) // 19
    for entry in action_log[-max_entries:]:
        if y + 19 > grid_h:
            break
        screen.blit(font_small.render(entry, True, BLACK), (x, y))
        y += 19


def _draw_frame():
    grid_w = game.GRID_COLS * CELL_SIZE
    grid_h = game.GRID_ROWS * CELL_SIZE
    screen.fill((215, 210, 198))

    for r in range(game.GRID_ROWS):
        for c in range(game.GRID_COLS):
            _draw_tile(r, c, game.layout[r][c])

    for item in game.world_items:
        _draw_item(item)

    _draw_chef()
    _draw_console()
    pygame.display.flip()


def _overlay_end(served: bool):
    """Draw a centred win/loss overlay on the grid area."""
    grid_w  = game.GRID_COLS * CELL_SIZE
    grid_h  = game.GRID_ROWS * CELL_SIZE
    msg     = "ORDER SERVED!" if served else "TIME'S UP!"
    colour  = (0, 160, 60) if served else (180, 40, 40)
    font_big = pygame.font.Font(None, 70)
    surf     = font_big.render(msg, True, colour)
    rect     = surf.get_rect(center=(grid_w // 2, grid_h // 2))
    bg_rect  = rect.inflate(22, 14)
    pygame.draw.rect(screen, WHITE, bg_rect, border_radius=8)
    pygame.draw.rect(screen, colour, bg_rect, 3, border_radius=8)
    screen.blit(surf, rect)
    font_sub = pygame.font.Font(None, 26)
    sub      = font_sub.render("Press R to restart", True, DARK_GRAY)
    screen.blit(sub, sub.get_rect(center=(grid_w // 2, grid_h // 2 + 50)))
    pygame.display.flip()


# ── Public API ────────────────────────────────────────────────────────────────

def setup(render: bool = True):
    """
    Create the environment and (optionally) open the pygame window.

    Call this once before main() or before your training loop.
    """
    global screen, clock, game, action_log, game_ended
    game       = OvercookedSimEnv(render_mode=None)
    game_ended = False
    action_log = ["Game started! Use keys to play."]
    game.reset()

    if render:
        pygame.init()
        grid_w = game.GRID_COLS * CELL_SIZE
        grid_h = game.GRID_ROWS * CELL_SIZE
        screen = pygame.display.set_mode((grid_w + CONSOLE_WIDTH, grid_h))
        pygame.display.set_caption("Overcooked Gym – Visualisation")
        clock  = pygame.time.Clock()


def main():
    """Interactive keyboard-controlled play loop."""
    global game_ended, action_log

    running = True
    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    game_ended = False
                    action_log.append("─── RESET ───")
                elif not game_ended:
                    if   event.key in (pygame.K_w, pygame.K_UP):    action = ACTION_UP
                    elif event.key in (pygame.K_s, pygame.K_DOWN):  action = ACTION_DOWN
                    elif event.key in (pygame.K_a, pygame.K_LEFT):  action = ACTION_LEFT
                    elif event.key in (pygame.K_d, pygame.K_RIGHT): action = ACTION_RIGHT
                    elif event.key == pygame.K_c:                   action = ACTION_CHOP
                    elif event.key in (pygame.K_e, pygame.K_SPACE): action = ACTION_PICKUP

        if action is not None and not game_ended:
            obs, reward, terminated, truncated, _ = game.step(action)
            entry = f"{ACTION_NAMES[action]}: r={reward:+.2f}"
            if terminated: entry += "  [SERVED!]"
            if truncated:  entry += "  [TIMEOUT]"
            action_log.append(entry)
            if len(action_log) > 60:
                action_log.pop(0)
            if terminated or truncated:
                game_ended = True

        _draw_frame()

        if game_ended:
            served = any("[SERVED!]" in e for e in action_log[-6:])
            _overlay_end(served)

        clock.tick(30)

    pygame.quit()
    sys.exit()


def refresh(obs, reward: float, terminated: bool, truncated: bool,
            info: dict, delay: float = 0.1):
    """
    Update the display from an external training / evaluation loop.

    Parameters
    ----------
    obs        : observation returned by env.step()
    reward     : reward returned by env.step()
    terminated : terminated flag returned by env.step()
    truncated  : truncated flag returned by env.step()
    info       : info dict; may contain 'action' (int) or 'action_name' (str)
    delay      : seconds to sleep after rendering (default 0.1)
    """
    global action_log, game_ended

    action_name = info.get('action_name') or ACTION_NAMES.get(info.get('action'), '?')
    entry = f"{action_name}: r={reward:+.2f}"
    if terminated: entry += "  [SERVED!]"
    if truncated:  entry += "  [TIMEOUT]"
    action_log.append(entry)
    if len(action_log) > 60:
        action_log.pop(0)

    if terminated or truncated:
        game_ended = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    _draw_frame()

    if game_ended:
        served = terminated  # truncated means timeout, not a serve
        _overlay_end(served)

    clock.tick(60)
    time.sleep(delay)


if __name__ == "__main__":
    setup()
    main()
