
import pickle
import csv
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
import numpy as np
from transformers import AutoTokenizer
import pdb
prefix = "data"
model_path = "./hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots/761d6a18cf99db371e0b43baf3e2d21b3e865a20"

tokenizer = AutoTokenizer.from_pretrained(model_path)
def smiles_to_2d_Morgan(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        includeChirality=True,
        fpSize=2048
    )
    fp = morgan_generator.GetFingerprint(mol)
    return fp

def process(name):
    with open(name + '.csv', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        columns = ['Reactant SMILES', 'Solvent SMILES', 'Catalyst SMILES', 'Label', 'Product SMILES','reaction_SMILES','Catalyst SMILES(RDKit)','target_val']
        dataset = []
        output_values = []
        for row in csvreader:
            filtered_row = {col: row[col] for col in columns}
            row['target_val'] = float(row['target_val'])
            output_values.append(row['target_val'])
            dataset += [filtered_row]
        # min_output = np.min(output_values)
        # max_output = np.max(output_values)
        min_output = -4.5
        max_output = 4.5
        print(min_output,max_output)
        max_length = 256
 
        for x, rea in enumerate(dataset):
            reactant_smiles = rea.get('Reactant SMILES', '')
            solvent_smiles = rea.get('Solvent SMILES', '')
            catalyst_smiles = rea.get('Catalyst SMILES(RDKit)', '')
            catalyst = rea.get('Catalyst SMILES', '')
            reactants_smiles = f"{reactant_smiles}.{solvent_smiles}.{catalyst_smiles}"
            # reactants_smiles = rea['reaction_SMILES'].split('>>')[0]
            rea['catalyst_2D'] = np.array(smiles_to_2d_Morgan(rea['Catalyst SMILES']))
            rea['solvent_2D'] = np.array(smiles_to_2d_Morgan(rea['Solvent SMILES']))
            rea['re_2D'] = np.array(smiles_to_2d_Morgan(rea['Reactant SMILES']))
            rea['Catalyst'] = rea['Catalyst SMILES']
            rea['Solvent']  = rea['Solvent SMILES'] 
            rea['Reactant'] = rea['Reactant SMILES']
            rea['Catalyst SMILES'] = tokenizer(rea['Catalyst SMILES'], padding='max_length', max_length=229, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Solvent SMILES'] = tokenizer(rea['Solvent SMILES'], padding='max_length', max_length=35, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Reactant SMILES'] = tokenizer(rea['Reactant SMILES'], padding='max_length', max_length=83, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Catalyst SMILES(RDKit)'] = tokenizer(rea['Catalyst SMILES(RDKit)'], padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Reactants'] = tokenizer(reactants_smiles, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Label'] = np.array(int(rea['Label']))
            rea['target_val'] = np.array((float(rea['target_val']) - min_output) / (max_output - min_output))
       

        with open(name + "_" + prefix  + '.pickle', 'wb') as file:
            pickle.dump(dataset, file)

        print(name, 'file saved.')
def process_thiol(name):
    with open(name + '.csv', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        columns = ['reaction', 'Output']
        dataset = []
        output_values = []
        for row in csvreader:
            filtered_row = {col: row[col] for col in columns}
          
            row['Output'] = float(row['Output'])
            output_values.append(row['Output'])
            dataset += [filtered_row]
        # min_output = np.min(output_values)
        # max_output = np.max(output_values)
        min_output = -0.419377753
        max_output = 3.134624572
        print(min_output,max_output)
        max_length = 256
        for x, rea in enumerate(dataset):
            rea['reaction'] = tokenizer(rea['reaction'], padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Output'] = np.array((float(rea['Output']) - min_output) / (max_output - min_output))
       
        

        with open(name + "_" + prefix + '.pickle', 'wb') as file:
            pickle.dump(dataset, file)

        print(name, 'file saved.')
def process_Rh(name):
    with open(name + '.csv', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        columns = ['Reactant SMILES', 'Solvent SMILES', 'Catalyst SMILES(RDKit)','ddG']
        dataset = []
        output_values = []
        for row in csvreader:
            filtered_row = {col: row[col] for col in columns}
            row['ddG'] = float(row['ddG'])
            output_values.append(row['ddG'])
            dataset += [filtered_row]
        min_output = np.min(output_values)
        max_output = np.max(output_values)
        print(min_output,max_output)
        max_length = 256
 
        for x, rea in enumerate(dataset):
            reactant_smiles = rea.get('Reactant SMILES', '')
            solvent_smiles = rea.get('Solvent SMILES', '')
            catalyst_smiles = rea.get('Catalyst SMILES(RDKit)', '')
            reactants_smiles = f"{reactant_smiles}.{solvent_smiles}.{catalyst_smiles}"
            rea['Solvent SMILES'] = tokenizer(rea['Solvent SMILES'], padding='max_length', max_length=35, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Reactant SMILES'] = tokenizer(rea['Reactant SMILES'], padding='max_length', max_length=83, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['Catalyst SMILES(RDKit)'] = tokenizer(rea['Catalyst SMILES(RDKit)'], padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['reaction'] = tokenizer(reactants_smiles, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids'].numpy().flatten()
            rea['ddG'] = np.array((float(rea['ddG']) - min_output) / (max_output - min_output))
        with open(name + "_" + prefix  + '.pickle', 'wb') as file:
            pickle.dump(dataset, file)

        print(name, 'file saved.')

if __name__ =='__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info') 
    # pdb.set_trace()
    # process("data/example_eval")
    # process_thiol("data/test_Catalyst")
    process_Rh("data/data3")