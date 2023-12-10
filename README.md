![LOVENet (3)](images/LOVENet.png)


# LOVENet: Large Optimized Vector Embeddings Network for Drug Repurposing

## Project Overview
A new framework maximizing the synergistic effects of knowledge graphs and large language models (LLMs) to discover novel therapeutic uses for pre-existing drugs. Specifically, our approach fuses information from pairs of embedding from Llama2 and heterogeneous knowledge graphs to reveal complex relations between drugs and diseases. 

## Project Structure
The repository comprises various Python scripts, each catering to a specific aspect of the LOVENet model:

- `LLM/representation.py`: Creating the embedding and feature representation from LLM.
- `GNN/main.py`: The main script that orchestrates the workflow of the model.
- `GNN/model.py`: Contains the architecture of the GNN and fusion with LLM representations.
- `GNN/layer.py`: Defines the custom layers used in the neural network.
- `GNN/utils.py`: Provides utility functions for data handling and auxiliary tasks.
- `GNN/args.py`: Handles parsing of command-line arguments.
- `GNN/load_data.py`: Loads and preprocesses the data for the model.

## Getting Started
First to extract the embedding of drug and disease run the following:
1. **LLM Embedding:**
   - Execute `representation.py` in `LLM/`.
   - Set inputs:
     - `path`: Directory ("B_data/").
     - `drug_disease_path`: Drug-disease relations file.
     - `disease_path`: Disease ID to name mapping file.
     - `drug_path`: Drug ID to name mapping file.
     - `directory_path`: Saved model directory.
    
2. **Run the main code**
First the following arguments should be specified:

```bash
python main.py --device_id 1 \
               --dataset /B-dataset \ # Target dataset
               --llm_rep_path /llm_rep_final_dataset.pkl \ # LLM representation extracted in step one
               --Disease_mapper /Disease_B_mapper.pkl \ # Disease ID to name mapping file
               --Drug_mapper /Drug_B_mapper.pkl \ # Drug ID to name mapping file
               --saved_path /B-dataset \
               --seed 0 \
               --print_every 1 \
               --nfold 5 \
               --epoch 200 \
               --batch_size 2048 \
               --learning_rate 0.005 \
               --weight_decay 0.0 \
               --check_metric auc \
               --k 15 \ # Top k similarities used as related drug disease
               --aggregate_type BiTrans \
               --topk 1 \ # Selecting k prediction result
               --hidden_feats 64 \
               --num_layer 2 \
               --dropout 0.2 \
               --batch_norm \
               --skip False \
               --mil False \
               --ins_predict False

```


### Prerequisites
- Python 3.10.12
- Ubuntu 22.04.1 LTS
- CUDA Version 12.0
- NVIDIA RTX 3090 Ti GPU

### Installing Dependencies
Run `pip install -r requirements.txt` to install the required Python libraries.



### Contact

For any queries regarding LOVENet, please contact the authors:

Sepehr Asgarian - sasgarian@klick.com
Sina Akbarian - sakbarian@klick.com
Jouhyun Jeon - cjeon@klick.com
