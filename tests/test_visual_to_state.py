"""
Test that _encode_state from the visual env produces a 30-float observation
vector that matches the format expected by hash_obs and the sim-trained Q-table.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'mdp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'mdp', 'gym'))

from Overcooked_mdp_gym import OvercookedEnv
from Overcooked_Q_Agent import hash_obs

# ------------------------------------------------------------------
# Slot labels matching _encode_state layout
# ------------------------------------------------------------------
SLOT_LABELS = [
    "[0]  chef_x",
    "[1]  chef_y",
    "[2]  held=none",
    "[3]  held=plate",
    "[4]  held=fish",
    "[5]  held=shrimp",
    "[6]  held=cutFish",
    "[7]  held=cutShrimp",
    "[8]  plate_x",
    "[9]  plate_y",
    "[10] fish_x",
    "[11] fish_y",
    "[12] shrimp_x",
    "[13] shrimp_y",
    "[14] cutFish_x",
    "[15] cutFish_y",
    "[16] cutShrimp_x",
    "[17] cutShrimp_y",
    "[18] facing=up",
    "[19] facing=down",
    "[20] facing=left",
    "[21] facing=right",
    "[22] order=cutFish",
    "[23] order=cutShrimp",
    "[24] plate_ing=none",
    "[25] plate_ing=plate",
    "[26] plate_ing=fish",
    "[27] plate_ing=shrimp",
    "[28] plate_ing=cutFish",
    "[29] plate_ing=cutShrimp",
]


def detect_order_from_image(img, env):
    """Run template matching against a still image instead of live screen grab."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use same region coords as _detect_order
    top, left = 0, 8
    h, w = 140, 116
    frame = gray[top:top + h, left:left + w]

    best_label = 0
    best_score = 0.6

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for label, path in [(1, os.path.join(root, "src", "templates", "fish_order.png")),
                        (2, os.path.join(root, "src", "templates", "shrimp_order.png"))]:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"  WARNING: template not found at {path}")
            continue
        if template.shape[0] >= frame.shape[0] or template.shape[1] >= frame.shape[1]:
            print(f"  WARNING: template larger than crop for {path}")
            continue
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_label = label

    return best_label


def test_image(image_path):
    print(f"\n{'=' * 60}")
    print(f"Image: {image_path}")
    print(f"{'=' * 60}")

    env = OvercookedEnv()
    env.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: could not load image at {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = env._detect_objects(img_rgb)

    print(f"Detections: {detections}")

    game_state = env._build_game_state(detections)

    print(f"Game state: {game_state}")

    order = detect_order_from_image(img, env)

    print(f"  Order detected: {order}  (0=none, 1=fish/cutFish, 2=shrimp/cutShrimp)")

    obs = env._encode_state(game_state, order)

    #print(f"\n  30-float observation vector:")
    for label, val in zip(SLOT_LABELS, obs):
        flag = " <-- one-hot active" if val == 1.0 else ""
        flag = " <-- UNDETECTED"     if val == -1.0 else flag
        print(f"    {label:28s} = {val:6.3f}{flag}")

    # Show what hash_obs produces — this is the actual Q-table key
    state_key = hash_obs(obs)
    key_labels = [
        "chef_x", "chef_y", "held", "facing",
        "plate_x", "plate_y", "fish_x", "fish_y",
        "shrimp_x", "shrimp_y", "cutFish_x", "cutFish_y",
        "cutShrimp_x", "cutShrimp_y", "order", "plate_ing"
    ]
    print(f"\n  Q-table key from hash_obs:")
    for label, val in zip(key_labels, state_key):
        print(f"    {label:15s} = {val}")

    # Basic sanity checks
    print(f"\n  Sanity checks:")
    checks = [
        ("obs is length 30",          len(obs) == 30),
        ("chef detected",             obs[0] != -1.0),
        ("held item has exactly one active slot",    np.sum(obs[2:8] == 1.0) == 1),
        ("facing has exactly one active slot",       np.sum(obs[18:22] == 1.0) == 1),
        ("plate_ing has exactly one active slot",    np.sum(obs[24:30] == 1.0) == 1),
        ("order slot valid",          np.max(obs[22:24]) >= 0),
    ]
    for desc, result in checks:
        print(f"    [{'PASS' if result else 'FAIL'}] {desc}")


# ------------------------------------------------------------------
# Run against available test images
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0000.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0102.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0103.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0104.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0105.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0106.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0107.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0186.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0187.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0188.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0189.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0190.png"))
test_image(os.path.join(ROOT, "src", "objectTrackingModels", "testImages", "frame_0191.png"))