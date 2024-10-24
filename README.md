
# Driving Through Graphs: A Bipartite Graph for Traffic Scene Analysis

## Description
This project introduces a novel approach for traffic scene analysis using frame-to-frame (f2f) bipartite graphs, which efficiently capture spatio-temporal relationships between objects in driving videos. The method simplifies the complexity typically associated with image-level high-dimensional feature extraction by representing 2D bounding boxes, object class and inferring euclidean information from it.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).
## METEOR Dataset Graphs Download
To download the constructed temporal bipartite graphs from the METEOR Dataset, which is used for training and evaluating the models, please use the following link:
[METEOR Dataset Download](https://drive.google.com/file/d/1T5PFb3iW6g8OnVl2SGuuB5Dqg9Ita2-h/view?usp=drive_link)

### Virtual Environment Setup (Recommended)
Recommended installation is Anaconda(https://www.anaconda.com/download)

```bash
# Install virtualenv if you haven't installed it yet
# Create a new Conda environment
conda create --name myenv python=3.8

# Activate the Conda environment
# On Windows
conda activate myenv
# On Unix or MacOS
conda activate myenv

# Install necessary packages
conda install pytorch torchvision torchaudio -c pytorch
conda install matplotlib
```

### Install Dependencies
Once your environment is set up and activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the code, navigate to the project directory and execute the `main.py` script with optional arguments if you wish to override the default configurations:

```bash
python main.py --train_directory path/to/train --test_directory path/to/test --num_epochs 100
```

You can adjust parameters such as `--batch_size`, `--learning_rate`, and others depending on your computational resources and requirements.
# Graph Visualization

This project includes a Python script `visualize_graphs.py` that facilitates the visualization of the temporal bipartite graph from `.pt` files using NetworkX and matplotlib. This script automatically loads graph data from `.pt` files located in a specified directory and displays each graph with a spring layout, providing a clear view of nodes and their connections.

### Usage

To use the visualization script, you need to update the `directory_path` in the script to the location where your `.pt` files are stored. After setting the correct path, run the script as follows:

```bash
python visualize_graphs.py
```

This will load each `.pt` file in the directory, convert it into a NetworkX graph, and display it using matplotlib, showing nodes, edges, and labels.
## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
- This research was partially funded by the UKRI EPSRC project ATRACT (EP/X028631/1): A Trustworthy Robotic Autonomous System for Casualty Triage.
