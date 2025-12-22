"""
config.py
Central repository for all stabilized parameters of the simulation.
"""

# --- World & Geometry Configuration ---
WORLD_SETTINGS = {
    "geometry": "square",       # Options: "square", "torus", "l-shape"
    "grid_size": (50, 50),      # Width, Height
    "transparency": 0.5,        # [0-1] Visibility of true history
    "fame_decay": 0.5,          # [0-1] Speed of reputation expiration
    "fame_radius": 5,           # Moore neighborhood radius for reputation
    "gossip_reliability": 0.5,   # [0-1] Accuracy of reputation transmission
    "initial_fame": 0.5        # Starting fame for unknown agents
}

# --- Game & Economic Physics ---
GAME_PHYSICS = {
    # PD Matrix: (Reward, Temptation, Sucker, Punishment)
    "payoff_matrix": {
        ("C", "C"): 4,
        ("D", "C"): 15,
        ("C", "D"): -12,
        ("D", "D"): 0
    },
    "base_existence_tax": 0.5,
    "cognitive_tax_rate": 0.01, # Cost per unit of memory capacity
    "movement_tax": 1,
    "interaction_cost": 0.5,    # Fee to engage in a game
    "migration_tax": 5        # Cost to colonize distant lands (Launch protocol)
}

# --- Population & Evolutionary Rules ---
POPULATION_SETTINGS = {
    "initial_agents": 500,
    "reproduction_threshold": 200,
    "mutation_rate": 0.1,
    "birth_protocol": "launch", # Options: "stay", "displace", "launch"
    "starting_points": 150,
    "max_age": 200             # Maximum lifespan in ticks
}

# --- Default DNA / Character Bounds ---
# These are the ranges used for initial generation and mutation
# --- Neural Network / Brain Configuration ---
BRAIN_SETTINGS = {
    "input_size": 6,      # [MyPts, MyAge, OppFame, OppHistory, Bias, Ideology]
    "hidden_size": 6,     # Neurons in hidden layer
    "output_size": 4,     # [Prob_Coop, Prob_Defect, Prob_Move, Prob_Ignore]
    "mutation_power": 0.2, # Standard deviation of Gaussian noise added to weights
    "mutation_rate": 0.1   # Probability of a weight being mutated
}