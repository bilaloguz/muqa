from simulation.engine import SimulationEngine
import numpy as np

def test_tribes():
    print("Starting Short Tribal stability test...")
    try:
        engine = SimulationEngine()
        print("Running 10 ticks...")
        for t in range(11):
            engine.run_tick()
            # Simulate what get_social_stats does
            pop = len(engine.agents)
            gen_sigs = np.array([a.dna["genetic_signature"] for a in engine.agents])
            cult_sigs = np.array([a.cultural_signature for a in engine.agents])
            avg_gen_div = np.mean(np.linalg.norm(gen_sigs - np.mean(gen_sigs, axis=0), axis=1))
            avg_cult_div = np.mean(np.linalg.norm(cult_sigs - np.mean(cult_sigs, axis=0), axis=1))
            avg_fame_fog = engine.total_fog / engine.interactions_this_tick if engine.interactions_this_tick > 0 else 0.0
            avg_kin_bias = np.mean([np.sum(np.abs(a.dna["W1"][5, :])) for a in engine.agents])
            avg_cult_bias = np.mean([np.sum(np.abs(a.dna["W1"][6, :])) for a in engine.agents])
            
            print(f"Tick {t}: Pop={pop} | GenDiv={avg_gen_div:.3f} | CultDiv={avg_cult_div:.3f} | Fog={avg_fame_fog:.3f} | KinBias={avg_kin_bias:.2f}")
                
        print("Short Tribal Identity test completed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_tribes()
