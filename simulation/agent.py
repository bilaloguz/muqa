"""
simulation/agent.py
Defines the Agent class with Neural Network (Brain) based decision making.
"""
import uuid
import random
import numpy as np
from config import POPULATION_SETTINGS, BRAIN_SETTINGS, GAME_PHYSICS, IDENTITY_SETTINGS

class Agent:
    def __init__(self, position, dna=None):
        # 1. Identity & State
        self.id = uuid.uuid4()
        self.points = POPULATION_SETTINGS["starting_points"]
        self.position = position
        self.age = random.randint(0, 50) 
        
        # 2. DNA (The Brain Weights & Traits)
        if dna is None:
            self.dna = self._init_brain()
        else:
            self.dna = dna
            
        # 3. Lifetime Learning State (The Plastic Layers)
        # These start as Zero and evolve during lifetime
        # Shape: Input Size -> Output Size (Direct parallel pathways)
        self.W_hebb = np.zeros((BRAIN_SETTINGS["input_size"], BRAIN_SETTINGS["output_size"]))
        self.W_rl = np.zeros((BRAIN_SETTINGS["input_size"], BRAIN_SETTINGS["output_size"]))
        
        # Learning Context
        self.last_input = None
        self.last_action = None
        
        # 4. Memory (Private Experience)
        self.private_memory = {}
        self.memory_capacity = int(self.dna["memory_capacity"])

        # 5. Cultural Identity (Fluid Status)
        # We start with the genetic baseline but it shifts during lifetime
        self.cultural_signature = self.dna["starting_culture"].copy()
        
    def _init_brain(self):
        """Initializes random weights and cognitive traits."""
        hidden_size = random.randint(BRAIN_SETTINGS["min_hidden"], BRAIN_SETTINGS["max_hidden"])
        mem_start = BRAIN_SETTINGS["min_memory"]
        mem_end = BRAIN_SETTINGS["max_memory"]
        
        # --- Tribal Signatures ---
        # Hardware (DNA)
        genetic_signature = np.random.randn(IDENTITY_SETTINGS["genetic_dim"])
        # Software (Starting Culture)
        starting_culture = np.random.uniform(0, 1, IDENTITY_SETTINGS["cultural_dim"])

        return {
            # --- Reptilian Layer (Static) ---
            "hidden_size": hidden_size,
            "W1": np.random.randn(BRAIN_SETTINGS["input_size"], hidden_size),
            "W2": np.random.randn(hidden_size, BRAIN_SETTINGS["output_size"]),
            
            # --- Cognitive Traits (Layer Weights) ---
            "w_reptilian": random.uniform(0.5, 1.5),
            "w_hebb": random.uniform(0.0, 1.0),
            "w_memetic": random.uniform(0.0, 1.0),
            "w_rl": random.uniform(0.0, 1.0),
            
            # --- Identity DNA ---
            "genetic_signature": genetic_signature,
            "starting_culture": starting_culture,
            
            # --- Personality Specs ---
            "creativity": random.uniform(0.01, 0.1), # Noise sigma
            "learning_rate": random.uniform(0.01, 0.1),
            "memory_capacity": random.randint(mem_start, mem_end)
        }

    def mutate(self):
        """Returns a mutated copy of the current DNA (Weights + Traits)."""
        new_dna = {
            "hidden_size": self.dna["hidden_size"],
            "W1": self.dna["W1"].copy(),
            "W2": self.dna["W2"].copy(),
            "memory_capacity": self.dna["memory_capacity"],
            
            # Traits
            "w_reptilian": self.dna["w_reptilian"],
            "w_hebb": self.dna["w_hebb"],
            "w_memetic": self.dna["w_memetic"],
            "w_rl": self.dna["w_rl"],
            "creativity": self.dna["creativity"],
            "learning_rate": self.dna["learning_rate"]
        }
        
        # --- Neurogenesis / Atrophy (Brain Resizing) ---
        if random.random() < BRAIN_SETTINGS["mutation_rate"]:
            current_h = new_dna["hidden_size"]
            choice = random.choice([-1, 1])
            new_h = max(BRAIN_SETTINGS["min_hidden"], min(BRAIN_SETTINGS["max_hidden"], current_h + choice))
            
            if new_h > current_h: # Growth: Add Neuron (Column to W1, Row to W2)
                # Add column to W1
                new_col = np.random.randn(BRAIN_SETTINGS["input_size"], 1)
                new_dna["W1"] = np.hstack((new_dna["W1"], new_col))
                # Add row to W2
                new_row = np.random.randn(1, BRAIN_SETTINGS["output_size"])
                new_dna["W2"] = np.vstack((new_dna["W2"], new_row))
                new_dna["hidden_size"] = new_h
                
            elif new_h < current_h: # Atrophy: Remove Neuron (Last Col of W1, Last Row of W2)
                # Remove last column of W1
                new_dna["W1"] = new_dna["W1"][:, :-1]
                # Remove last row of W2
                new_dna["W2"] = new_dna["W2"][:-1, :]
                new_dna["hidden_size"] = new_h

        # Apply Standard Noise Mutation
        # Apply Gaussian noise to W1
        mask1 = np.random.rand(*new_dna["W1"].shape) < BRAIN_SETTINGS["mutation_rate"]
        noise1 = np.random.randn(*new_dna["W1"].shape) * BRAIN_SETTINGS["mutation_power"]
        new_dna["W1"][mask1] += noise1[mask1]
        
        # Apply Gaussian noise to W2
        mask2 = np.random.rand(*new_dna["W2"].shape) < BRAIN_SETTINGS["mutation_rate"]
        noise2 = np.random.randn(*new_dna["W2"].shape) * BRAIN_SETTINGS["mutation_power"]
        new_dna["W2"][mask2] += noise2[mask2]
        
        # --- Trait Mutation ---
        for trait in ["w_reptilian", "w_hebb", "w_memetic", "w_rl", "creativity", "learning_rate"]:
            if random.random() < BRAIN_SETTINGS["mutation_rate"]:
                noise = random.uniform(-0.1, 0.1)
                new_dna[trait] += noise
                new_dna[trait] = max(0.0, new_dna[trait])

        # Mutate Memory Capacity
        if random.random() < BRAIN_SETTINGS["mutation_rate"]:
            change = random.randint(-2, 2)
            new_mem = new_dna["memory_capacity"] + change
            new_dna["memory_capacity"] = max(BRAIN_SETTINGS["min_memory"], min(BRAIN_SETTINGS["max_memory"], new_mem))
            
        # --- Identity Mutation ---
        new_dna["genetic_signature"] = self.dna["genetic_signature"].copy()
        if random.random() < IDENTITY_SETTINGS["mutation_rate"]:
            noise = np.random.randn(IDENTITY_SETTINGS["genetic_dim"]) * 0.1
            new_dna["genetic_signature"] += noise

        # Inherit Culture (Cultural Transmission from Parent's Current State)
        new_dna["starting_culture"] = self.cultural_signature.copy()
        if random.random() < IDENTITY_SETTINGS["mutation_rate"]:
            noise = np.random.uniform(-0.1, 0.1, IDENTITY_SETTINGS["cultural_dim"])
            new_dna["starting_culture"] += noise
            new_dna["starting_culture"] = np.clip(new_dna["starting_culture"], 0, 1)
        
        return new_dna

    def update_memory(self, opponent_id, move):
        """Adds a move to private memory."""
        if opponent_id not in self.private_memory:
            self.private_memory[opponent_id] = []
        
        history = self.private_memory[opponent_id]
        history.append(move)
        
        if len(history) > self.memory_capacity:
            history.pop(0)

    def update_culture(self, opponent_move, opponent_culture):
        """
        Updates the agent's internal cultural identity.
        Mutual cooperation leads to hybridization; betrayal leads to polarization.
        """
        if opponent_move == "C": # Hybridization
            rate = IDENTITY_SETTINGS["hybridization_rate"]
            # Shift towards neighbor
            diff = opponent_culture - self.cultural_signature
            self.cultural_signature += diff * rate
        elif opponent_move == "D": # Polarization
            rate = IDENTITY_SETTINGS["polarization_rate"]
            # Shift away from neighbor
            diff = opponent_culture - self.cultural_signature
            self.cultural_signature -= diff * rate
        
        self.cultural_signature = np.clip(self.cultural_signature, 0, 1)

    def decide(self, opponent_id, opponent_fame, neighbors=None):
        """
        The Layered Brain Forward Pass.
        Combines Instinct, Habit, Social Pressure, Value, and Creativity.
        """
        # --- 1. PREPARE INPUTS ---
        in_points = min(self.points / 1000.0, 1.0)
        in_age = min(self.age / POPULATION_SETTINGS["max_age"], 1.0)
        in_fame = opponent_fame
        
        in_history = 0.5
        if opponent_id in self.private_memory:
            hist = self.private_memory[opponent_id]
            if hist:
                in_history = hist.count("C") / len(hist)
        
        in_bias = 1.0
        
        # --- Tribal Proximity Inputs ---
        # 1. Kinship Proximity (Genetic Similarity)
        # Using 1 / (1 + Euclidean Distance) for a 0-1 range
        gen_dist = np.linalg.norm(self.dna["genetic_signature"] - opponent_id.dna["genetic_signature"]) if hasattr(opponent_id, 'dna') else 2.0
        in_kin_prox = 1.0 / (1.0 + gen_dist)
        
        # 2. Cultural Proximity
        cult_dist = np.linalg.norm(self.cultural_signature - opponent_id.cultural_signature) if hasattr(opponent_id, 'cultural_signature') else 1.0
        in_cult_prox = 1.0 / (1.0 + cult_dist)
        
        # Shape: (1, 7)
        inputs = np.array([in_points, in_age, in_fame, in_history, in_bias, in_kin_prox, in_cult_prox])
        
        # --- 2. MULTI-LAYER PROCESSING ---
        logits = np.zeros(BRAIN_SETTINGS["output_size"])
        
        # Layer 1: Reptilian (Instinct) - DNA Static
        # ReLU(Input @ W1) @ W2
        hidden = np.maximum(np.dot(inputs, self.dna["W1"]), 0)
        reptilian_logits = np.dot(hidden, self.dna["W2"])
        logits += self.dna["w_reptilian"] * reptilian_logits

        # Layer 2: Hebbian (Habit) - Association
        # Input @ W_hebb
        hebbian_logits = np.dot(inputs, self.W_hebb)
        logits += self.dna["w_hebb"] * hebbian_logits

        # Layer 3: Reinforcement (Value) - Q-Learning-ish
        # Input @ W_rl
        rl_logits = np.dot(inputs, self.W_rl)
        logits += self.dna["w_rl"] * rl_logits

        # Layer 4: Memetic (Social) - Copying Success
        # Average of neighbors' last moves, weighted by their wealth relative to me
        # We assume neighbors have a 'last_action_index' property public
        if neighbors and self.dna["w_memetic"] > 0:
            social_vector = np.zeros(BRAIN_SETTINGS["output_size"])
            valid_neighbors = 0
            for n in neighbors:
                if n.points > self.points and hasattr(n, 'last_action_index') and n.last_action_index is not None:
                    # One-hot encoding of their move
                    move_vec = np.zeros(BRAIN_SETTINGS["output_size"])
                    move_vec[n.last_action_index] = 1.0
                    social_vector += move_vec
                    valid_neighbors += 1
            
            if valid_neighbors > 0:
                social_vector /= valid_neighbors
                logits += self.dna["w_memetic"] * social_vector

        # Layer 5: Perturbative (Creativity) - Noise
        noise = np.random.randn(BRAIN_SETTINGS["output_size"]) * self.dna["creativity"]
        logits += noise
        
        # --- 3. SELECTION ---
        action_index = np.argmax(logits)
        choices = ["C", "D", "MOVE", "IGNORE"]
        
        # --- 4. STORAGE FOR LEARNING ---
        self.last_input = inputs
        self.last_action_index = action_index
        
        return choices[action_index]

    def learn(self, reward):
        """
        Updates the Plastic Layers (Hebbian & RL) based on the outcome.
        """
        if self.last_input is None or self.last_action_index is None:
            return

        # Hebbian Update: Strengthen connection between State and Action
        # W_hebb += rate * (ActionVec * InputVec)
        action_vec = np.zeros(BRAIN_SETTINGS["output_size"])
        action_vec[self.last_action_index] = 1.0
        
        # Outer product to get matrix of changes
        hebbian_delta = np.outer(self.last_input, action_vec) # Shape: 6x4 (Input x Output)
        # Note: self.W_hebb should be (6, 4). My init was (Inputs, Outputs). 
        # But np.dot(inputs, W) implies W is (Inputs, Outputs). 
        # Correct check: Input is (6,), W is (6,4). np.dot -> (4,). Correct.
        # np.outer(6, 4) -> (6,4). Correct.
        
        self.W_hebb += self.dna["learning_rate"] * hebbian_delta

        # RL Update: Reward Modulated
        # W_rl += rate * Reward * (ActionVec * InputVec)
        # We assume reward is centered around 0 for better stability, or we just take raw.
        # Payoff matrix has [-10, +10]. 
        rl_delta = hebbian_delta * reward
        self.W_rl += self.dna["learning_rate"] * rl_delta
        
        # Normalize to prevent explosion? 
        # Optional: Decay
        self.W_hebb *= 0.99
        self.W_rl *= 0.99

    def is_alive(self):
        return self.points > 0

    def __repr__(self):
        return f"<Agent {self.id.hex[:4]} | Age: {self.age} | Pts: {self.points:.1f}>"