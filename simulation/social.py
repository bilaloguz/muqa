"""
simulation/social.py
Manages the global 'Social Fame' ledger, reputation decay, and gossip.
"""
from config import WORLD_SETTINGS
import random

class SocialLedger:
    def __init__(self):
        # The master record of every agent's public actions.
        # Format: { agent_id: {'C': total_cooperations, 'D': total_defections, 'history': []} }
        self.registry = {}
        
        self.decay_rate = WORLD_SETTINGS["fame_decay"]
        self.transparency = WORLD_SETTINGS["transparency"]
        self.initial_fame = WORLD_SETTINGS["initial_fame"]

    def register_agent(self, agent_id):
        """Initializes a new agent in the social records."""
        if agent_id not in self.registry:
            self.registry[agent_id] = {
                "C": 0,
                "D": 0,
                "history": [] # Recent moves for decay calculation
            }

    def record_action(self, agent_id, action):
        """Records a public action (C or D) for an agent."""
        if agent_id not in self.registry:
            self.register_agent(agent_id)
        
        self.registry[agent_id][action] += 1
        self.registry[agent_id]["history"].append(action)

    def get_fame(self, agent_id):
        """
        Calculates the 'Social Fame' of an agent.
        Returns a value between 0 (Pure Defector) and 1 (Pure Cooperator).
        """
        if agent_id not in self.registry:
            return self.initial_fame  # Use the custom social bias
        
        data = self.registry[agent_id]
        total = data["C"] + data["D"]
        
        if total == 0:
            return 0.5
            
        # Raw Reputation: C / Total
        raw_reputation = data["C"] / total
        
        # Apply Transparency Logic:
        # If transparency is 0.5, the 'perceived' fame is shifted toward neutral (0.5)
        perceived_fame = 0.5 + (raw_reputation - 0.5) * self.transparency

        # Apply Gossip Reliability (Noise)
        # 1.0 = No Noise, 0.0 = High Noise
        reliability = WORLD_SETTINGS.get("gossip_reliability", 1.0)
        if reliability < 1.0:
            noise_range = (1.0 - reliability)
            noise = random.uniform(-noise_range, noise_range)
            perceived_fame += noise
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, perceived_fame))

    def apply_fame_decay(self):
        """
        Periodically reduces the weight of old actions.
        This allows agents to 'redeem' themselves over time.
        """
        for agent_id in self.registry:
            # We multiply existing counts by (1 - decay_rate)
            # e.g., if decay is 0.1, counts drop by 10% each tick.
            self.registry[agent_id]["C"] *= (1 - self.decay_rate)
            self.registry[agent_id]["D"] *= (1 - self.decay_rate)