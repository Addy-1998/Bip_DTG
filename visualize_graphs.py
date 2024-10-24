
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def load_and_visualize_graph(file_path):
    # Load graph data
    data = torch.load(file_path)

    # Convert to NetworkX graph
    graph = to_networkx(data, to_undirected=True)

    # Visualize the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)  # Layout algorithm to determine node positions
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=100)
    nx.draw_networkx_edges(graph, pos, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title(f"Graph Visualization of {os.path.basename(file_path)}")
    plt.axis('off')
    plt.show()

# Example usage: Visualize graphs from a directory
directory_path = 'path_to_your_pt_files'  # Update this path to your .pt files directory
for filename in os.listdir(directory_path):
    if filename.endswith('.pt'):
        file_path = os.path.join(directory_path, filename)
        load_and_visualize_graph(file_path)
