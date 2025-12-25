from simulation.engine import SimulationEngine
import numpy as np

def test_tribes():
    print("Starting Tribal Identity stability test...")
    try:
        engine = SimulationEngine()
        # Check if agents have signatures
        a = engine.agents[0]
        if not hasattr(a, 'cultural_signature') or len(a.cultural_signature) != 3:
            raise ValueError("Agent missing valid cultural signature.")
        if "genetic_signature" not in a.dna or len(a.dna["genetic_signature"]) != 3:
            raise ValueError("Agent missing valid genetic signature.")
            
        print("Initial checks passed. Running ticks...")
        for t in range(100):
            engine.run_tick()
            if t % 20 == 0:
                # Find an agent with some history
                target = next((a for a in engine.agents if a.id in engine.social_ledger.registry), engine.agents[1])
                observer = engine.agents[0]
                
                fame_true = engine.social_ledger.get_fame(None, target)
                fame_perc = engine.social_ledger.get_fame(observer, target)
                
                pop = len(engine.agents)
                print(f"Tick {t}: Pop={pop} | Target={target.id.hex[:4]} | True={fame_true:.2f} | Perc={fame_perc:.2f}")
                
        print("Tribal Identity test completed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_tribes()
