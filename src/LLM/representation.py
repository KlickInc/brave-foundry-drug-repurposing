import pandas as pd
import os
import torch
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration for Bits and Bytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", quantization_config=bnb_config, device_map={"":1})

def llamagenerator(question):
    """
    Generates an answer from the LLM model based on the provided question.
    
    :param question: A string containing the question for the LLM.
    :return: Numpy array representing the logits from the model output.
    """
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    output = model(input_ids)
    return output.logits[0,-1,:].cpu().detach().numpy()

def process_data(drug_disease_path, disease_path, drug_path, directory_path):
    """
    Processes the drug-disease data and generates model outputs.

    :param drug_disease_path: Path to the drug-disease CSV file.
    :param disease_path: Path to the disease CSV file.
    :param drug_path: Path to the drug CSV file.
    :param mapper_drug_path: Path to the drug mapping CSV file.
    :param directory_path: Path where the output file will be saved.
    """
    # Load data
    drug_disease = pd.read_csv(drug_disease_path)
    disease = pd.read_csv(disease_path).rename(columns={"CUI_Disease_name_umls_simple":"name"})
    drug = pd.read_csv(drug_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Generate LLM answers and save to dictionary
    dict_p = {}
    with tqdm(total=drug_disease.shape[0] * drug_disease.shape[1]) as pbar:
        pbar.set_description('Processing Drug-Disease pairs')
        for drug_id in range(drug_disease.shape[0]):
            for disease_id in range(drug_disease.shape[1]):
                drug_name = drug.iloc[drug_id]['name']
                disease_name = disease.iloc[disease_id]['name']
                question = f"Is this drug: {drug_name} related to this disease: {disease_name} answer yes or no ?"
                answer_llm = llamagenerator(question)
                dict_key = f"{drug_id}_{drug_name}_{disease_id}_{disease_name}"
                dict_p[dict_key] = answer_llm
                pbar.update(1)

    # Save the generated data
    dict_file_path = os.path.join(directory_path, 'llm_rep_final_dataset.pkl')
    with open(dict_file_path, 'wb') as file:
        pickle.dump(dict_p, file)
    print('Data prepared and saved to', dict_file_path)

# Example usage for a specific dataset
# Process the first dataset (B-dataset)
#path to the data 
path = "/B_data/" 
process_data(
    drug_disease_path= path + "drug_disease.csv",
    disease_path= path + "disease.csv",
    drug_path= path + "drug.csv",
    directory_path= "/saved_model"
)
