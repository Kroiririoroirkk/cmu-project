import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_neurons = 10
eps = 0.05

rng = np.random.default_rng()
M = np.abs(rng.normal(size=num_neurons*(num_neurons-1)//2))
G = nx.complete_graph(num_neurons)

fig = plt.figure(figsize=(8,8))
pos = nx.circular_layout(G)
nodes = nx.draw_networkx_nodes(G, pos)
edges = nx.draw_networkx_edges(G, pos, width=M)

def update(n):
    edges.set(linewidth=M*(M>=n*eps))
    return edges,

anim = FuncAnimation(fig, update, frames=50, interval=100, blit=True)
anim.save('anim.gif')
