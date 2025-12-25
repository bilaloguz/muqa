"""
main.py
The orchestration script. Runs the simulation, visualizes it, and logs history.
"""
import time
import sys
import numpy as np
from simulation.engine import SimulationEngine
from utils.visualizer import Visualizer
from utils.logger import WorldLogger
from config import WORLD_SETTINGS

def get_social_stats(engine):
    pop = len(engine.agents)
    if pop == 0:
        return None

    import numpy as np
    
    avg_points = sum(a.points for a in engine.agents) / pop
    avg_fame = sum(engine.social_ledger.get_fame(observer=None, target=a) for a in engine.agents) / pop
    avg_mem = sum(a.memory_capacity for a in engine.agents) / pop
    avg_cult = np.mean([a.cultural_signature for a in engine.agents], axis=0)
    avg_hidden = sum(a.dna["hidden_size"] for a in engine.agents) / pop
    
    # --- New Brain Metrics ---
    # 1. Cognitive Stack (Avg Trust Weights)
    avg_w_reptilian = sum(a.dna["w_reptilian"] for a in engine.agents) / pop
    avg_w_hebb = sum(a.dna["w_hebb"] for a in engine.agents) / pop
    avg_w_memetic = sum(a.dna["w_memetic"] for a in engine.agents) / pop
    avg_w_rl = sum(a.dna["w_rl"] for a in engine.agents) / pop
    
    # 2. Wisdom (Matrix Norms)
    # How "trained" are they?
    avg_hebb_norm = sum(np.linalg.norm(a.W_hebb) for a in engine.agents) / pop
    avg_rl_norm = sum(np.linalg.norm(a.W_rl) for a in engine.agents) / pop
    
    # 3. Creativity
    avg_creativity = sum(a.dna["creativity"] for a in engine.agents) / pop
    
    # --- 4. Tribal Metrics ---
    # Divergence: Variance/Distance from mean
    gen_sigs = np.array([a.dna["genetic_signature"] for a in engine.agents])
    cult_sigs = np.array([a.cultural_signature for a in engine.agents])
    
    avg_gen_div = np.mean(np.linalg.norm(gen_sigs - np.mean(gen_sigs, axis=0), axis=1))
    avg_cult_div = np.mean(np.linalg.norm(cult_sigs - np.mean(cult_sigs, axis=0), axis=1))
    
    # Social Fog (Sampled from engine tracking)
    avg_fame_fog = engine.total_fog / engine.interactions_this_tick if engine.interactions_this_tick > 0 else 0.0
    
    # Identity Bias (Absolute weights of identity inputs in Reptilian layer)
    # Inputs: [..., KinProx, CultProx] at indices 5 and 6
    avg_kin_bias = np.mean([np.sum(np.abs(a.dna["W1"][5, :])) for a in engine.agents])
    avg_cult_bias = np.mean([np.sum(np.abs(a.dna["W1"][6, :])) for a in engine.agents])

    return {
        "tick": engine.tick,
        "pop": pop,
        "avg_pts": avg_points,
        "avg_fame": avg_fame,
        "avg_mem": avg_mem,
        "avg_idl": np.mean(avg_cult),
        "avg_hidden": avg_hidden,
        
        # Cognitive Stack
        "avg_w_reptilian": avg_w_reptilian,
        "avg_w_hebb": avg_w_hebb,
        "avg_w_memetic": avg_w_memetic,
        "avg_w_rl": avg_w_rl,
        
        # Wisdom & Creativity
        "avg_hebb_norm": avg_hebb_norm,
        "avg_rl_norm": avg_rl_norm,
        "avg_creativity": avg_creativity,
        
        # Tribal Landscape
        "avg_gen_div": avg_gen_div,
        "avg_cult_div": avg_cult_div,
        "avg_fame_fog": avg_fame_fog,
        "avg_kin_bias": avg_kin_bias,
        "avg_cult_bias": avg_cult_bias,
        
        "total_C": engine.coops_this_tick,
        "total_D": engine.defects_this_tick,
        "total_deaths": engine.deaths_this_tick
    }

def main():
    print("--- MUQA SIMULATION STARTING ---")
    engine = SimulationEngine()
    viz = Visualizer(WORLD_SETTINGS["grid_size"])
    logger = WorldLogger()
    
    print(f"Logging to: {logger.get_log_path()}")

    MAX_TICKS = 5000
    try:
        for t in range(MAX_TICKS):
            # 1. Run Engine
            engine.run_tick()
            
            # 2. Handle Statistics
            stats = get_social_stats(engine)
            
            # 3. Update Visualization
            if t % 1 == 0: # Smooth 60fps
                viz.update(engine.world, engine.social_ledger, t, stats)
            
            # 3. Handle Statistics & Logging
            stats = get_social_stats(engine)
            if stats:
                logger.log_tick(stats)
                if t % 5 == 0:
                    status = (f"Tick: {t:04d} | Pop: {stats['pop']:03d} | "
                             f"Avg Fame: {stats['avg_fame']:.2f} | Pts: {stats['avg_pts']:.1f}")
                    sys.stdout.write("\r" + status)
                    sys.stdout.flush()
            else:
                print("\nSocietal Extinction Reached.")
                break
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    finally:
        print("\nFinalizing logs...")
        viz.close()

if __name__ == "__main__":
    main()