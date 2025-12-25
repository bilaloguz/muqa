"""
utils/visualizer.py
The "Eye" of the simulation. Renders the social grid with wealth and reputation data.
Updated to strictly use the Red-Yellow-Green (RdYlGn) diverging spectrum.
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Force interactive backend for Linux environments
try:
    matplotlib.use('TkAgg') 
except:
    pass

class Visualizer:
    def __init__(self, world_size):
        self.width, self.height = world_size
        plt.ion() 
        
        # Setup Figure with GridSpec: 4 Columns
        # Col 0: Map (Wide)
        # Col 1: Macro Social Stats
        # Col 2: Advanced Brain Profile
        # Col 3: Tribal Landscape
        self.fig = plt.figure(figsize=(24, 12))
        self.fig.canvas.manager.set_window_title('Muqa Simulation')
        gs = self.fig.add_gridspec(3, 4, width_ratios=[1.5, 1, 1, 1])
        
        # --- Column 0: Map ---
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        
        # --- Column 1: Macro Statistics ---
        self.ax_pop = self.fig.add_subplot(gs[0, 1])
        self.ax_social = self.fig.add_subplot(gs[1, 1])
        self.ax_wealth = self.fig.add_subplot(gs[2, 1])
        
        # --- Column 2: Brain Profile ---
        self.ax_struct = self.fig.add_subplot(gs[0, 2])
        self.ax_wisdom = self.fig.add_subplot(gs[1, 2])
        self.ax_stack = self.fig.add_subplot(gs[2, 2])
        
        # --- Column 3: Tribal Landscape ---
        self.ax_div = self.fig.add_subplot(gs[0, 3])
        self.ax_fog = self.fig.add_subplot(gs[1, 3])
        self.ax_bias = self.fig.add_subplot(gs[2, 3])
        
        self.cbar = None
        
        # Data History
        self.history_ticks = []
        self.history_pop = []
        self.history_fame = []
        self.history_idl = []
        self.history_wealth = []
        self.history_mem = []
        self.history_hidden = []
        
        # Cognitive Stack Data
        self.hist_w_reptilian = []
        self.hist_w_hebb = []
        self.hist_w_memetic = []
        self.hist_w_rl = []
        
        # Wisdom & Creativity
        self.hist_hebb_norm = []
        self.hist_rl_norm = []
        self.hist_creativity = []
        
        # Tribal Landscape
        self.history_gen_div = []
        self.history_cult_div = []
        self.history_fame_fog = []
        self.history_kin_bias = []
        self.history_cult_bias = []

    def _add_legend(self):
        """Creates a custom legend for dot sizes (Wealth/Points)."""
        sizes = [20, 100, 200]
        labels = ["Poor", "Middle", "Wealthy"]
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=labels[i],
                   markerfacecolor='gray', markersize=np.sqrt(sizes[i]), 
                   markeredgecolor='black')
            for i in range(len(sizes))
        ]
        #self.ax_map.legend(handles=legend_elements, loc='upper right', title="Wealth Levels")

    def update(self, world, social_ledger, tick, stats=None):
        """Redraws the world state and updates graphs."""
        
        # --- 1. Draw Map ---
        self.ax_map.clear()
        self.ax_map.set_title(f"Tick: {tick} | Pop: {stats['pop']} | Social Map", fontsize=14)
        self.ax_map.set_xlim(-0.5, self.width - 0.5)
        self.ax_map.set_ylim(-0.5, self.height - 0.5)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.1)

        x_coords, y_coords, sizes, colors, edge_colors = [], [], [], [], []

        for x in range(self.width):
            for y in range(self.height):
                agent = world.grid[x, y]
                if agent is not None:
                    x_coords.append(x)
                    y_coords.append(y)
                    # Culture Vector -> RGB (Fill Color)
                    base_color = agent.cultural_signature
                    colors.append(base_color)
                    
                    # Genetic Vector -> RGB (Edge Color)
                    # Shift and clip to ensure valid [0, 1] RGB
                    gen_color = np.clip((agent.dna["genetic_signature"] + 2) / 4, 0, 1)
                    edge_colors.append(gen_color)
                    
                    size = min(max(agent.points / 2, 10), 400)
                    sizes.append(size)

        if x_coords:
            scatter = self.ax_map.scatter(
                x_coords, y_coords, 
                c=colors, s=sizes, 
                edgecolors=edge_colors, linewidths=1.5, alpha=0.9
            )
            
            # Map is now multi-colored by tribe, colorbar removed
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None

        self._add_legend()

        # --- 2. Update Graphs ---
        if stats:
            self.history_ticks.append(tick)
            self.history_pop.append(stats['pop'])
            self.history_fame.append(stats['avg_fame'])
            self.history_idl.append(stats.get('avg_idl', 0.5))
            self.history_wealth.append(stats.get('avg_pts', 0))
            self.history_mem.append(stats.get('avg_mem', 10.0))
            self.history_hidden.append(stats.get('avg_hidden', 6.0))
            
            # Stack Data
            self.hist_w_reptilian.append(stats['avg_w_reptilian'])
            self.hist_w_hebb.append(stats['avg_w_hebb'])
            self.hist_w_memetic.append(stats['avg_w_memetic'])
            self.hist_w_rl.append(stats['avg_w_rl'])
            
            # Wisdom Data
            self.hist_hebb_norm.append(stats['avg_hebb_norm'])
            self.hist_rl_norm.append(stats['avg_rl_norm'])
            self.hist_creativity.append(stats['avg_creativity'])
            
            # Tribal Data
            self.history_gen_div.append(stats.get('avg_gen_div', 0))
            self.history_cult_div.append(stats.get('avg_cult_div', 0))
            self.history_fame_fog.append(stats.get('avg_fame_fog', 0))
            self.history_kin_bias.append(stats.get('avg_kin_bias', 0))
            self.history_cult_bias.append(stats.get('avg_cult_bias', 0))

            # === COLUMN 1: MACRO STATISTICS ===
            
            # Chart 1.1: Population
            self.ax_pop.clear()
            self.ax_pop.plot(self.history_ticks, self.history_pop, color='#3498db', linewidth=1.5)
            self.ax_pop.set_title("Population Dynamics")
            self.ax_pop.grid(True, alpha=0.3)
            
            # Chart 1.2: Social Capital
            self.ax_social.clear()
            self.ax_social.plot(self.history_ticks, self.history_fame, color='#27ae60', linewidth=1.5, label='Avg Fame')
            self.ax_social.plot(self.history_ticks, self.history_idl, color='#8e44ad', linewidth=1.5, linestyle='--', label='Avg Ideology')
            self.ax_social.set_title("Social Capital")
            self.ax_social.set_ylim(0, 1)
            self.ax_social.legend(loc='lower right', fontsize='x-small')
            self.ax_social.grid(True, alpha=0.3)
            
            # Chart 1.3: Average Wealth
            self.ax_wealth.clear()
            self.ax_wealth.plot(self.history_ticks, self.history_wealth, color='#f39c12', linewidth=1.5)
            self.ax_wealth.set_title("Economic Prosperity (Avg Pts)")
            self.ax_wealth.grid(True, alpha=0.3)
            
            # === COLUMN 2: BRAIN PROFILE ===
            
            # Chart 2.1: Brain Structure (Hardware)
            self.ax_struct.clear()
            self.ax_struct.plot(self.history_ticks, self.history_hidden, color='#e74c3c', linewidth=1.5, label='Neurons')
            self.ax_struct.plot(self.history_ticks, self.history_mem, color='#3498db', linewidth=1.5, linestyle='--', label='Memory')
            self.ax_struct.set_title("Cognitive Hardware")
            self.ax_struct.legend(loc='upper left', fontsize='x-small')
            self.ax_struct.grid(True, alpha=0.3)

            # Chart 2.2: Wisdom (Software)
            self.ax_wisdom.clear()
            self.ax_wisdom.plot(self.history_ticks, self.hist_hebb_norm, color='#2ecc71', linestyle='-', label='Habit L2')
            self.ax_wisdom.plot(self.history_ticks, self.hist_rl_norm, color='#e67e22', linestyle='-', label='Value L2')
            self.ax_wisdom.set_title("Cognitive Software")
            self.ax_wisdom.legend(loc='upper left', fontsize='x-small')
            self.ax_wisdom.grid(True, alpha=0.3)
            
            # Chart 2.3: Cognitive Influence (Execution)
            self.ax_stack.clear()
            
            # Scale creativity to influence weight
            inf_rep = np.array(self.hist_w_reptilian)
            inf_heb = np.array(self.hist_w_hebb)
            inf_rl = np.array(self.hist_w_rl)
            inf_mem = np.array(self.hist_w_memetic)
            inf_cre = np.array(self.hist_creativity) * 10.0 
            
            total = inf_rep + inf_heb + inf_rl + inf_mem + inf_cre
            total[total == 0] = 1.0
            
            stack_data = np.row_stack((inf_rep/total, inf_heb/total, inf_rl/total, inf_mem/total, inf_cre/total))
            
            latest = stack_data[:, -1]
            labels = [
                f'Instinct: {latest[0]*100:.1f}%',
                f'Habit: {latest[1]*100:.1f}%',
                f'Value: {latest[2]*100:.1f}%',
                f'Social: {latest[3]*100:.1f}%',
                f'Noise: {latest[4]*100:.1f}%'
            ]
            
            self.ax_stack.stackplot(self.history_ticks, stack_data, 
                                    labels=labels,
                                    colors=['#34495e', '#f1c40f', '#e74c3c', '#3498db', '#9b59b6'], 
                                    alpha=0.8)
            
            self.ax_stack.set_title("Cognitive Execution Mix (%)")
            self.ax_stack.set_ylim(0, 1)
            self.ax_stack.legend(loc='lower left', fontsize='xx-small', ncol=2, framealpha=0.5)
            self.ax_stack.grid(True, alpha=0.1)

            # === COLUMN 3: TRIBAL LANDSCAPE ===
            
            # Chart 3.1: Tribal Divergence
            self.ax_div.clear()
            self.ax_div.plot(self.history_ticks, self.history_gen_div, color='#95a5a6', label='Genetic Var')
            self.ax_div.plot(self.history_ticks, self.history_cult_div, color='#1abc9c', label='Cultural Var')
            self.ax_div.set_title("Tribal Divergence")
            self.ax_div.legend(loc='upper left', fontsize='x-small')
            self.ax_div.grid(True, alpha=0.3)
            
            # Chart 3.2: Social Fog (Perception Error)
            self.ax_fog.clear()
            self.ax_fog.plot(self.history_ticks, self.history_fame_fog, color='#c0392b', linewidth=2)
            self.ax_fog.set_title("Social Fog (Perc. Error)")
            self.ax_fog.set_ylim(0, 0.5)
            self.ax_fog.grid(True, alpha=0.3)
            
            # Chart 3.3: Identity Priority (Weights)
            self.ax_bias.clear()
            self.ax_bias.plot(self.history_ticks, self.history_kin_bias, color='#2c3e50', label='Kinship')
            self.ax_bias.plot(self.history_ticks, self.history_cult_bias, color='#16a085', label='Cultural')
            self.ax_bias.set_title("Identity Priority (Instinct)")
            self.ax_bias.legend(loc='upper left', fontsize='x-small')
            self.ax_bias.grid(True, alpha=0.3)

        # Render
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Cleanup after simulation ends."""
        plt.ioff()
        plt.show()