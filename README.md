![LOVENet (3)](https://github.com/KlickInc/brave-foundry-drug-repurposing/assets/38491713/cfca5dee-89de-4817-9e0a-a4a943665dd1)


# LOVENet: Large Optimized Vector Embeddings Network for Drug Repurposing

## Project Overview
A new framework maximizing the synergistic effects of knowledge graphs and large language models (LLMs) to discover novel therapeutic uses for pre-existing drugs. Specifically, our approach fuses information from pairs of embedding from Llama2 and heterogeneous knowledge graphs to derive complex relations of drugs and diseases. 

## Project Structure
The repository comprises various Python scripts, each catering to a specific aspect of the LOVENet model:

- `representation.py`: Creating the embedding and feature representation from LLM.
- `main.py`: The main script that orchestrates the workflow of the model.
- `model.py`: Contains the architecture of the GNN and fusion with LLM representations.
- `layer.py`: Defines the custom layers used in the neural network.
- `utils.py`: Provides utility functions for data handling and auxiliary tasks.
- `args.py`: Handles parsing of command-line arguments.
- `load_data.py`: Loads and preprocesses the data for the model.

## Getting Started


### Prerequisites



### Contact

For any queries regarding LOVENet, please contact the authors:

Sepehr Asgarian - sasgarian@klick.com
Sina Akbarian - sakbarian@klick.com
Jouhyun Jeon - cjeon@klick.com
