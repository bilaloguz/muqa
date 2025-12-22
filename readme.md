# PD-Sim: Neuroevolutionary Prisoner's Dilemma

An advanced **Agent-Based Simulation (ABS)** exploring the evolution of cooperation, culture, and intelligence in a competitive digital ecosystem. This project models how Neural Networks (`Brain`), inheritable culture (`Ideology`), and economic pressure (`Taxes`) shape the emergence of complex societies.

---

## 🇬🇧 English Documentation

### 1. Project Overview
This simulation places autonomous agents on a 2D grid where they interact via the **Prisoner's Dilemma** game. Unlike simple rule-based models, these agents possess evolving **Neural Networks** that learn strategies over generations. They reproduce, migrate, form cultures, and eventually die, allowing "Survival of the Fittest" to select for the most successful behavioral traits.

### 2. Core Mechanics

#### 🧠 The Brain (Neuroevolution)
Every agent decides its actions using a Feed-Forward Neural Network (3-Layer Perceptron):
*   **Inputs (6):**
    1.  `MyPoints`: Current wealth (Survival pressure).
    2.  `MyAge`: Current age (Life expectancy).
    3.  `OppFame`: The opponent's public reputation.
    4.  `OppHistory`: Personal memory of this opponent.
    5.  `Bias`: Constant input.
    6.  **`Ideology`:** Internal state representing "Willingness to Cooperate" (Trust).
*   **Outputs (4):** `[Cooperate, Defect, Move, Ignore]`.
*   **Evolution:** Offspring inherit their parent's weight matrices (`W1`, `W2`) with Gaussian noise mutation (`mutation_power`).

#### 🧬 Genetics & Culture
Traits are no longer just hardcoded probabilities; they are complex biological and cultural markers:
*   **Ideology Inheritance:** Children inherit their parent's *current* Ideology (Optimism/Cynicism). This creates "Cultural Momentum"—a high-trust parent raises a high-trust child.
*   **Evolvable Memory:** `memory_capacity` is a gene (5-50 slots). Smarter agents pay higher taxes. Nature decides if "High IQ" is worth the metabolic cost.

#### ⚖️ Social Dynamics
*   **Ideology (Trust Model):** Agents have a "Mood" (0.0=Cynic, 1.0=Idealist). Cooperation boosts this mood; betrayal destroys it. This acts as a short-term Recurrent Memory.
*   **Gossip Reliability:** Public fame is noisy. Agents might hear misinformation (`gossip_reliability`), simulating the "Fog of War" in social reputation.
*   **Migration:** If an area is overcrowded, rich agents can pay a **Migration Tax** (`migration_tax`) to "Launch" their offspring to a distant, empty part of the world.

#### 💰 Economic Physics
The world is governed by strict thermodynamic laws:
*   **Payoff Matrix:** High temptation to defect (`15` points) vs moderate reward for cooperation (`4`).
*   **Taxes:**
    *   **Existence Tax:** Cost of breathing (`base_existence_tax`).
    *   **Cognitive Tax:** Cost of being smart (`cognitive_tax_rate` * Memory).
    *   **Movement Tax:** Cost of running away.
    *   **Migration Tax:** Cost of colonizing new lands.

### 3. File Structure
*   `main.py`: Entry point. Runs the visualizer and game loop.
*   `config.py`: **The Control Room.** Tune taxes, mutation rates, and game physics here.
*   `simulation/`:
    *   `agent.py`: The Neural Network and genetic logic.
    *   `engine.py`: The Physics Engine (Movement, Death, Reproduction).
    *   `social.py`: The Reputation System.
*   `verify_fix.py`: Headless script for fast statistical verification.

### 4. How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run Visualization (The Movie)
python3 main.py

# Run Statistical Analysis (The Data)
python3 verify_fix.py
```

---

## 🇹🇷 Türkçe Dokümantasyon

### 1. Proje Özeti
Bu simülasyon, "Tutsak İkilemi" oyunu üzerinden etkileşime giren ajanları modeller. Basit kurallar yerine, ajanlar nesiller boyu öğrenen **Sinir Ağlarına (Yapay Zeka)** sahiptir. Ürerler, göç ederler, kültür oluştururlar ve ölürler; böylece "Doğal Seçilim" en başarılı davranışları belirler.

### 2. Temel Mekanikler

#### 🧠 Beyin (Nöroevrim)
Her ajan kararlarını bir Sinir Ağı ile verir:
*   **Girdiler (6):** Puanım, Yaşım, Rakibin Şöhreti, Özel Hafızam, Bias, **İdeoloji**.
*   **Çıktılar (4):** `[İşbirliği, İhanet, Kaç, Yoksay]`.
*   **Evrim:** Çocuklar, ebeveynlerinin beyin ağırlıklarını (`W1`, `W2`) mutasyonla miras alır.

#### 🧬 Genetik ve Kültür
*   **İdeoloji Mirası:** Çocuklar, ebeveynlerinin *o anki* dünya görüşünü (İyimserlik/Kötümserlik) miras alır. Bu, "Güven Kültürü"nün nesiller boyu aktarılmasını sağlar.
*   **Evrimleşen Hafıza:** Hafıza kapasitesi (Zeka) bir gendir. Zeki ajanlar daha yüksek "Bilişsel Vergi" öder. Doğa, zekanın maliyetine değip değmeyeceğine karar verir.

#### 💰 Ekonomik Fizik
*   **Göç (Migration):** Zengin ajanlar, kalabalık bölgelerden kaçmak için **Göç Vergisi** ödeyerek çocuklarını dünyanın uzak köşelerine "Fırlatabilir" (Colonization).
*   **Bilişsel Vergi:** Zeki olmak pahalıdır (`cognitive_tax_rate`). Aptal ve tasarruflu olmak bazen daha iyi bir stratejidir.

### 3. Nasıl Çalıştırılır
```bash
python3 main.py
```
*Görselleştirme penceresi açılır. Evrimi izleyin.*