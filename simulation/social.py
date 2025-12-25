"""
simulation/social.py
Manages the global 'Social Fame' ledger, reputation decay, and gossip.
"""
import random
import numpy as np
from config import WORLD_SETTINGS

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

    def get_fame(self, observer, target):
        """
        Calculates the 'Relational Fame' of a target as perceived by an observer.
        Fame is distorted by physical distance (Geography) and Identity bias.
        """
        target_id = target.id
        if target_id not in self.registry:
            return self.initial_fame
        
        data = self.registry[target_id]
        total = data["C"] + data["D"]
        
        if total == 0:
            return 0.5
            
        # 1. Base Truth
        base_reputation = data["C"] / total
        
        if observer is None:
            return base_reputation

        # 2. Geographic Filter (Physical Distance)
        import numpy as np
        dist = np.linalg.norm(np.array(observer.position) - np.array(target.position))
        
        # News fades as distance increases relative to fame_radius
        geo_clarity = 1.0
        if dist > WORLD_SETTINGS["fame_radius"]:
            # Drop clarity exponentially outside the radius
            geo_clarity = np.exp(-(dist - WORLD_SETTINGS["fame_radius"]) / 5.0)
            
        # 3. Identity Filter (Tribal Integrity)
        # 3a. Genetic Proximity -> Data Integrity (Noise reduction)
        gen_dist = np.linalg.norm(observer.dna["genetic_signature"] - target.dna["genetic_signature"])
        kin_prox = 1.0 / (1.0 + gen_dist)
        
        # 3b. Cultural Proximity -> Data Network (Propagation/Clarity)
        cult_dist = np.linalg.norm(observer.cultural_signature - target.cultural_signature)
        cult_prox = 1.0 / (1.0 + cult_dist)
        
        # Combined Clarity: Influenced by geography and culture
        clarity = self.transparency * geo_clarity * (0.5 + 0.5 * cult_prox)
        
        # Perceived Fame shifts towards neutral (0.5) as clarity drops
        perceived_fame = 0.5 + (base_reputation - 0.5) * clarity
        
        # 4. Identity Bias (Noise)
        # Noise is maximized for strangers (Low KinProx)
        max_bias = WORLD_SETTINGS.get("identity_gossip_bias", 0.4)
        noise_range = max_bias * (1.0 - kin_prox)
        
        if noise_range > 0:
            import random
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