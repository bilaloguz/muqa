"""
config.py
Central repository for all stabilized parameters of the simulation.
"""

# --- World & Geometry Configuration ---
WORLD_SETTINGS = {
    "geometry": "square",       # Options: "square", "torus", "l-shape"
    "grid_size": (50, 50),      # Width, Height
    "transparency": 0.8,       # Base shared information quality
    "fame_decay": 0.05,        # How fast reputation is forgotten
    "fame_radius": 5,         # Base geography for news travel
    "gossip_reliability": 0.9, # Global channel quality
    "identity_gossip_bias": 0.4,# Maximum distortion caused by being a stranger
    "initial_fame": 0.5,       # Starting neutral reputation
}

# --- Game & Economic Physics ---
GAME_PHYSICS = {
    # PD Matrix: (Reward, Temptation, Sucker, Punishment)
    "payoff_matrix": {
        ("C", "C"): 5,
        ("D", "C"): 10,
        ("C", "D"): -5,
        ("D", "D"): -1
    },
    "base_existence_tax": 0.1,
    "cognitive_tax_rate": 0.01, # Cost per unit of memory capacity
    "brain_complexity_tax": 0.02, # Cost per hidden neuron
    "movement_tax": 1,
    "interaction_cost": 0.01,     # Fee to engage in a game
    "migration_tax": 10
}

# --- Population & Evolutionary Rules ---
POPULATION_SETTINGS = {
    "initial_agents": 500,
    "reproduction_threshold": 200,
    "mutation_rate": 0.1,
    "birth_protocol": "displace", # Options: "stay", "displace", "launch"
    "starting_points": 150,
    "max_age": 100             # Maximum lifespan in ticks
}

# --- Identity & Tribal Configuration ---
IDENTITY_SETTINGS = {
    "genetic_dim": 3,          # Dimensions of the "Green Beard" vector
    "cultural_dim": 3,         # Dimensions of the "Flag" vector
    "mutation_rate": 0.05,     # Probability of identity vector drift
    "hybridization_rate": 0.05, # Shift toward partner's culture on cooperation
    "polarization_rate": 0.1  # Shift away from partner's culture on betrayal
}

# --- Default DNA / Character Bounds ---
# These are the ranges used for initial generation and mutation
# --- Neural Network / Brain Configuration ---
BRAIN_SETTINGS = {
    "input_size": 7,      # [MyPts, MyAge, OppFame, OppHistory, Bias, KinProx, CultProx]
    "output_size": 4,     # [C, D, MOVE, IGNORE]
    "min_hidden": 2,
    "max_hidden": 16,
    "min_memory": 5,
    "max_memory": 50,
    "mutation_power": 0.2, # Standard deviation of Gaussian noise added to weights
    "mutation_rate": 0.1   # Probability of a weight being mutated
}

# --- Brain Layer Metabolic Costs ---
# Each cognitive layer has an associated metabolic cost per tick
BRAIN_COSTS = {
    "reptilian": 0.01,     # Base cost for the neural network
    "hebbian": 0.01,       # Cost of habit-based learning
    "memetic": 0.02,       # Cost of social observation/imitation
    "reinforcement": 0.03, # Cost of reward-based learning
    "creative": 0.01,      # Cost of random exploration/noise
    "identity": 0.005      # Cost of processing tribal signatures
}