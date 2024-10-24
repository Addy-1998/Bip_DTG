
# Driving Through Graphs: A Bipartite Graph for Traffic Scene Analysis
[Read the Paper](https://ieeexplore.ieee.org/document/10647492)
![Demo GIF](images/main.gif)
We introduce a novel approach for traffic scene analysis in driving videos by exploring spatio-temporal relationships captured by a temporal frame-to-frame (f2f) bipartite graph, eliminating the need for complex image-level high-dimensional feature extraction. Instead, we rely on object detectors that provide bounding box information. The proposed graph approach efficiently connects objects across frames where nodes represent essential object attributes, and edges signify interactions based on simple spatial metrics such as distance and angles between objects. A key innovation is the integration of dynamic edge attributes, computed using Multilayer Perceptrons (MLP) by exploring this spatial metric. These attributes enhance our Interaction-aware Graph Neural Networks (IA-GNNs) framework by adapting the PageRank-driven approximate personalized propagation of neural predictions (APPNP) scheme and graph attention mechanism in a novel way. This has significantly improved our model’s ability to understand spatio-temporal interactions of multiple objects in traffic scenarios. We have rigorously evaluated our approach on two benchmark datasets, METEOR and INTERACTION, demonstrating its accuracy in analyzing traffic scenarios. This streamlined, graph-based strategy marks a significant shift towards more efficient and insightful traffic scene analysis using video data

# Results

## Performance Comparison
Our model outperforms state-of-the-art methods on both accuracy and mean average precision (mAP) on the [METEOR Dataset](https://gamma.umd.edu/researchdirections/autonomousdriving/meteor/) and the [INTERACTION Dataset](https://interaction-dataset.com/).

![Results Table](images/Capture.PNG)
## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
- This research was partially funded by the UKRI EPSRC project ATRACT (EP/X028631/1): A Trustworthy Robotic Autonomous System for Casualty Triage.
