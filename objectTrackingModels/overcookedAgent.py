# List all class objects that can be tracked
CLASS_NAMES = {
    0: "chef",
    1: "plate",
    2: "fish",
    3: "shrimp",
    4: "cutFish",
    5: "cutShrimp"
}

# Beginning of a game state based on observations

game_state = {
    # location of the chef
    "chef": None,
    # location of the plates
    "plates": [],
    # location/type of food
    "food": []
}