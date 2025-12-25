# Muqa: An Evolutionary Simulation of Social "Intelligence"

An advanced **Multi-Agent Simulation (MAS)** exploring the co-evolution of social strategies, cognitive architectures, and tribal identities in a competitive digital ecosystem governed by game theory and metabolic economics.

---

## 🇬🇧 Technical Overview

### 1. The Core Philosophy
Muqa models a society of autonomous agents on a 2D grid where survival is tied to their ability to play the **Iterated Prisoner's Dilemma**. Unlike traditional models, these agents don't have hard-coded strategies. Their behavior emerges from a **5-Layer Cognitive Stack** and a **Vector-Based Identity System**.

### 2. Cognitive Architecture: The 5-Layer Brain
Every agent possesses a modular neural processor where distinct layers compete to influence the final decision (Cooperate, Defect, Move or Ignore).

*   **Layer 1: Reptilian (Instinct)**: A static, evolved MLP defined by the agent's DNA. It provides the "baseline" behavior for specific inputs.
*   **Layer 2: Hebbian (Habit)**: An associative learning layer that strengthens connections between recurring observations and outcomes.
*   **Layer 3: Reinforcement (Value)**: A Q-learning-inspired layer that builds a matrix of rewards/punishments based on personal history.
*   **Layer 4: Memetic (Social)**: An imitation layer that observes wealthy neighbors and "votes" for their successful behaviors (Prestige Bias).
*   **Layer 5: Perturbative (Creativity/Noise)**: Adds Gaussian exploration noise to the decision stack, allowing for "Aha!" moments and preventing stagnation.

**The "Cognitive Execution Mix"**: Evolution adjusts the *weighted reliance* on these layers. Some species survive by pure instinct (Reptilian), while others survive by cold calculation (RL) or social conformity (Memetic).

### 3. Tribal Identity: Genes vs. Memes
Identity is not binary; it is represented by 3-dimensional vectors.
*   **Genetic Signature (The Green Beard)**: A fixed DNA array (visualized as agent **Borders**). It represents immutable biological markers.
*   **Cultural Signature (The Flag)**: A mutable array (visualized as agent **Fill Color**). It is inherited from the parent's *current* state and shifts throughout life.
    *   **Hybridization**: Mutual cooperation nudges cultural vectors closer.
    *   **Polarization**: Betrayal causes cultural vectors to drift apart.

### 4. Relational Gossip & Social Fog
Reputation is no longer a global "God View." Status is subjective:
*   **Fame Radius**: News of actions propagates exponentially slower as physical distance increases.
*   **Data Integrity**: Genetic proximity reduces gossip noise. You hear the "truth" about your kin.
*   **Data Network**: Cultural proximity increases signal strength/clarity. You are more aware of the "deeds" of those who share your "Flag."
*   **Social Fog**: The delta between reality and perception. In tribal societies, the fog is thick between different groups, allowing bad actors to hide in the "mist" of stranger-hood.

### 5. Metabolic Economy
Every action and thought has a price (`BRAIN_COSTS`):
*   **Hardware Tax**: Agents pay per hidden neuron and per unit of memory capacity.
*   **Software Tax**: Active learning (Hebbian, RL, Memetic) requires constant metabolic energy.
*   **Existence Tax**: A fixed "burn rate" to stay alive.
*   **Reproduction**: Agents must exceed a wealth threshold (`reproduction_threshold`) to spawn offspring.

### 6. The 4-Column Dashboard
1.  **Map View**: Real-time rendering of tribes. (Fill = Culture, Border = Genetics).
2.  **Macro Stats**: Population counts, Societal Wealth, and Social Capital (Fame/Ideology).
3.  **Brain Profile**: Tracking the evolution of "Neurons" and the "Execution Mix" (which layer is driving the bus).
4.  **Tribal Landscape**: Visualizing the "Social Fog" (perception error) and "Identity Priority" (which biases the species is evolving).

---

## 🇹🇷 Teknik Özet

### 1. Temel Felsefe
Muqa, 'Tutsak İkilemi' (Prisoner's Dilemma) oyununu temel alan, ajanların sosyal stratejilerini ve zihinsel yeteneklerini evrimleştirdiği gelişmiş bir **Çoklu Ajan Simülasyonudur (MAS)**. Ajanların davranışları sabit kodlanmamış; **5 Katmanlı Zihinsel Yığın** ve **Vektör Tabanlı Kimlik Sistemi** üzerinden evrimsel süreçte şekillenmiştir.

### 2. Zihinsel Mimari: 5 Katmanlı Zihin
Her ajan, farklı katmanların kararlarını (Yardımlaş, İhanet Et, Hareket Et veya Görmezden Gel) etkilemek için yarıştığı modüler bir "zihin"e sahiptir:

*   **1. Katman: Sürüngen (İçgüdü)**: DNA tarafından tanımlanan statik bir sinir ağıdır. Temel davranışları belirler.
*   **2. Katman: Hebbian (Alışkanlık)**: Sürekli tekrarlanan olaylar arasında ilişki kuran çağrışımsal öğrenme katmanıdır.
*   **3. Katman: Pekiştirmeli (Değer)**: Kişisel deneyimlere dayalı ödül/ceza matrisi oluşturan Q-öğrenme tabanlı katmandır.
*   **4. Katman: Memetik (Sosyal)**: Başarılı komşuları gözlemleyen ve onların stratejilerini taklit eden katmandır (Prestij Yanlılığı).
*   **5. Katman: Yaratıcı (Gürültü)**: Ajanın keşif yapmasını sağlayan ve yerinde saymasını engelleyen rastgele gürültü katmanıdır.

**Bilişsel İcra Karışımı**: Evrim, bu katmanlara olan güveni ayarlar. Bazı türler saf içgüdüyle hayatta kalırken, bazıları sosyal uyumla hayatta kalır.

### 3. Kabile Kimliği: Genler ve Memler
Kimlik ikili değil, 3 boyutlu vektörlerle temsil edilir:
*   **Genetik İmza (Yeşil Sakal)**: Sabit DNA dizisidir (Ajanların **Kenar Rengi**). Değişmez biyolojik markerları temsil eder.
*   **Kültürel İmza (Bayrak)**: Değişebilir kültürel dizidir (Ajanların **İç Rengi**). Ebeveynden miras alınır ve yaşam boyunca şekillenir.
    *   **Hibritleşme**: Karşılıklı yardımlaşma, kültürel vektörleri birbirine yaklaştırır.
    *   **Kutuplaşma**: İhanet, kültürel vektörlerin birbirinden uzaklaşmasına neden olur.

### 4. İlişkisel Dedikodu ve Sosyal Sis
İtibar artık küresel değil, özneldir:
*   **Ün Yarıçapı**: Haberler, fiziksel mesafe arttıkça katlanarak daha yavaş yayılır.
*   **Veri Bütünlüğü**: Genetik yakınlık, dedikodu gürültüsünü azaltır. Kendi akrabalarınız hakkındaki gerçeği daha net duyarsınız.
*   **Sosyal Sis**: Gerçek ile algı arasındaki farktır. Kabileci toplumlarda yabancılar arasındaki sis kalındır, bu da kötü aktörlerin "yabancılık sisi" arkasında saklanmasına olanak tanır.

### 5. Metabolik Ekonomi
Her eylemin ve düşüncenin bir bedeli vardır (`BRAIN_COSTS`):
*   **Donanım Vergisi**: Ajanlar sahip oldukları her nöron ve hafıza birimi için vergi öderler.
*   **Yazılım Vergisi**: Aktif öğrenme katmanları (RL, Hebbian vb.) sürekli metabolik enerji gerektirir.
*   **Üreme**: Ajanlar, yavrulamak için belli bir servet eşiğini (`reproduction_threshold`) geçmek zorundadır.

### 6. 4 Sütunlu Kontrol Paneli
1.  **Harita Görünümü**: Kabilelerin gerçek zamanlı görünümü (İç Renk = Kültür, Kenar = Genetik).
2.  **Makro İstatistikler**: Nüfus, toplumsal zenginlik ve sosyal sermaye.
3.  **Bilişsel Profil**: Nöron evrimi ve hangi bilişsel katmanın baskın olduğunun takibi.
4.  **Kabile Manzarası**: "Sosyal Sis" (algı hatası) ve "Kimlik Önceliği"nin (genetik vs kültürel önyargı) görselleştirilmesi.

---

## 🛠️ Installation & Running

```bash
# Clone the repository
git clone https://github.com/your-repo/muqa.git
cd muqa

# Install dependencies (Numpy, Matplotlib)
pip install -r requirements.txt

# Start the simulation
python3 main.py
```

## ⚙️ Configuration Reference (`config.py`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `fame_radius` | 15 | Distance news travels spatially. |
| `identity_gossip_bias`| 0.4 | Distortion caused by tribal unfamiliarity. |
| `hybridization_rate` | 0.05 | Rate of cultural convergence on cooperation. |
| `polarization_rate` | 0.1 | Rate of cultural divergence on betrayal. |
| `reproduction_threshold`| 200 | Wealth required to reproduce. |
| `brain_complexity_tax`| 0.02 | Energy cost per hidden neuron per tick. |
