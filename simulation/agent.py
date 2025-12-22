"""
simulation/agent.py
Defines the Agent class with Neural Network (Brain) based decision making.
"""
import uuid
import random
import numpy as np
from config import POPULATION_SETTINGS, BRAIN_SETTINGS, GAME_PHYSICS

class Agent:
    def __init__(self, position, dna=None):
        # 1. Identity & State
        self.id = uuid.uuid4()
        self.points = POPULATION_SETTINGS["starting_points"]
        self.position = position
        self.age = random.randint(0, 50) # Random start to prevent mass extinction
        
        # 2. DNA (The Brain Weights)
        # If no parents, random initialization
        if dna is None:
            self.dna = self._init_brain()
        else:
            self.dna = dna
            
        # 3. Memory (Private Experience) - Still used as input for the brain
        self.private_memory = {}
        # Dynamic memory capacity from DNA
        self.memory_capacity = int(self.dna["memory_capacity"])

        # 4. Ideology (State Variable: Willingness to Cooperate)
        # 0.0 = Cynic, 1.0 = Idealist
        # Inherit from DNA or start random neutral-ish
        self.ideology = self.dna.get("starting_ideology", random.uniform(0.3, 0.7))
        
    def _init_brain(self):
        """Initializes random weights and memory traits."""
        return {
            "W1": np.random.randn(BRAIN_SETTINGS["input_size"], BRAIN_SETTINGS["hidden_size"]),
            "W2": np.random.randn(BRAIN_SETTINGS["hidden_size"], BRAIN_SETTINGS["output_size"]),
            "memory_capacity": random.randint(5, 20),
            "starting_ideology": random.uniform(0.3, 0.7)
        }

    def mutate(self):
        """Returns a mutated copy of the current DNA (Weights + Traits)."""
        new_dna = {
            "W1": self.dna["W1"].copy(),
            "W2": self.dna["W2"].copy(),
            "memory_capacity": self.dna["memory_capacity"]
        }
        
        # Apply Gaussian noise to W1
        mask1 = np.random.rand(*new_dna["W1"].shape) < BRAIN_SETTINGS["mutation_rate"]
        noise1 = np.random.randn(*new_dna["W1"].shape) * BRAIN_SETTINGS["mutation_power"]
        new_dna["W1"][mask1] += noise1[mask1]
        
        # Apply Gaussian noise to W2
        mask2 = np.random.rand(*new_dna["W2"].shape) < BRAIN_SETTINGS["mutation_rate"]
        noise2 = np.random.randn(*new_dna["W2"].shape) * BRAIN_SETTINGS["mutation_power"]
        new_dna["W2"][mask2] += noise2[mask2]
        
        # Mutate Memory Capacity
        if random.random() < BRAIN_SETTINGS["mutation_rate"]:
            change = random.randint(-2, 2)
            new_mem = new_dna["memory_capacity"] + change
            new_dna["memory_capacity"] = max(1, min(50, new_mem))
            
        # Inherit Ideology (Cultural Transmission from Parent's Current State)
        # We start with the PARENT'S CURRENT IDEOLOGY, not their birth ideology
        inherited_ideology = self.ideology
        
        # Mutate Ideology
        if random.random() < BRAIN_SETTINGS["mutation_rate"]:
            noise = random.uniform(-0.1, 0.1)
            inherited_ideology += noise
            
        new_dna["starting_ideology"] = max(0.0, min(1.0, inherited_ideology))
        
        return new_dna

    def update_memory(self, opponent_id, move):
        """Adds a move to private memory."""
        if opponent_id not in self.private_memory:
            self.private_memory[opponent_id] = []
        
        history = self.private_memory[opponent_id]
        history.append(move)
        
        if len(history) > self.memory_capacity:
            history.pop(0)

    def update_ideology(self, opponent_move):
        """
        Updates the agent's internal willingness to cooperate (Ideology).
        Trust Model: Cooperation builds trust (Ideology Up), Defection breaks it (Ideology Down).
        """
        # Learning Rate / Volatility
        decay = 0.9 
        
        target = 0.5 # Default neutral pull
        if opponent_move == "C":
            target = 1.0 # Reinforce Optimism
        elif opponent_move == "D":
            target = 0.0 # Reinforce Cynicism
        
        # Apply Update: Ideology shifts towards the target
        self.ideology = (self.ideology * decay) + (target * (1 - decay))

    def decide(self, opponent_id, opponent_fame):
        """
        The Neural Forward Pass.
        Inputs: [MyPoints(norm), MyAge(norm), OppFame, OppHistory, Bias, Ideology]
        """
        # --- 1. PREPARE INPUTS ---
        # Normalize points (0-1 range approx, capped at 1000)
        in_points = min(self.points / 1000.0, 1.0)
        
        # Normalize age
        in_age = min(self.age / POPULATION_SETTINGS["max_age"], 1.0)
        
        # Opponent Reputation (0-1)
        in_fame = opponent_fame
        
        # Opponent Personal History (0=Defector, 1=Cooperator, 0.5=Unknown)
        in_history = 0.5
        if opponent_id in self.private_memory:
            hist = self.private_memory[opponent_id]
            if hist:
                in_history = hist.count("C") / len(hist)
        
        # Bias Input
        in_bias = 1.0
        
        # Ideology Input
        in_ideology = self.ideology
        
        inputs = np.array([in_points, in_age, in_fame, in_history, in_bias, in_ideology])
        
        # --- 2. FORWARD PROPAGATION ---
        # Layer 1: Inputs -> Hidden (ReLU)
        hidden = np.dot(inputs, self.dna["W1"])
        hidden = np.maximum(hidden, 0) # ReLU
        
        # Layer 2: Hidden -> Output
        output = np.dot(hidden, self.dna["W2"])
        
        # Softmax (optional, or just argmax)
        # For simple decision, we just check which neuron is highest
        # Output[0] = Cooperate
        # Output[1] = Defect
        # Output[2] = Move
        # Output[3] = Ignore
        
        choices = ["C", "D", "MOVE", "IGNORE"]
        action_index = np.argmax(output)
        
        return choices[action_index]

    def is_alive(self):
        return self.points > 0

    def __repr__(self):
        return f"<Agent {self.id.hex[:4]} | Age: {self.age} | Pts: {self.points:.1f}>"