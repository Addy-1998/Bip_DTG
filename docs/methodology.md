
# Methodology

## Problem Definition
The goal is to classify events in driving videos using a graph-based approach to model multi-object interactions. Our approach processes traffic scene data with minimal reliance on pixel-level features.

## Graph Construction
We use frame-to-frame bipartite graphs to represent temporal object interactions across consecutive frames. Each graph is constructed with objects as nodes and relationships between objects as edges.

## Interaction Aware GNN (IA-GNN)
We employ an Interaction Aware Graph Neural Network to process the constructed graphs. The network utilizes dynamic edge attributes, which are calculated using multilayer perceptrons (MLPs), to predict events in traffic scenes.
