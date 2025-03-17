import torch
from torch.utils.data import Dataset
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
import pdb


class TransformerDataset(Dataset):
    """

    """
    def __init__(self, if_shuffle, data):
        # (L, B, C)
        self.data = data # list of dict of tensors
        self.shuffle = if_shuffle
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        for feature_dict in data:
            for key in feature_dict:
                if key in list_smiles:
                    feature_dict[key] = feature_dict[key]
                elif key == 'target_val' or key == 'double':
                    tmp = torch.tensor(feature_dict[key], dtype=torch.float)
                    feature_dict[key] = tmp
                else:
                    tmp = torch.tensor(feature_dict[key])     
                    feature_dict[key] = tmp.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'catalyst_2D': data['catalyst_2D'],
            'solvent_2D': data['solvent_2D'],
            're_2D': data['re_2D'],
            'Catalyst SMILES': data['Catalyst SMILES'],
            'Solvent SMILES': data['Solvent SMILES'],
            'Reactant SMILES': data['Reactant SMILES'],
            'Catalyst SMILES(RDKit)': data['Catalyst SMILES(RDKit)'],
            'Label': data['Label'],
            'target_val': data['target_val'],
            'double': data['double'],
            'Catalyst': data['Catalyst'],
            'Solvent': data['Solvent'],
            'Reactant': data['Reactant'],
            'Product SMILES': data['Product SMILES'],
            'Reactants' : data['Reactants'],
            'reaction_SMILES' : data['reaction_SMILES']
        }
    
    def collate_fn(data_list):

        batch = {}
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        for key in data_list[0]:
            lst = [i[key] for i in data_list]
            if key in list_smiles:
                batch[key] = lst
            else:   
                batch[key] = torch.stack(lst)
        
        return batch

class TransformerDataset_Thiol(Dataset):
    """

    """
    def __init__(self, if_shuffle, data):
        # (L, B, C)
        self.data = data # list of dict of tensors
        self.shuffle = if_shuffle
        for feature_dict in data:
            for key in feature_dict:
                if key == 'Output':
                    tmp = torch.tensor(feature_dict[key])     
                    feature_dict[key] = tmp
                else:
                    tmp = torch.tensor(feature_dict[key])     
                    feature_dict[key] = tmp.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'reaction': data['reaction'],
            'Output': data['Output']
        }
    
    def collate_fn(data_list):

        batch = {}
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        for key in data_list[0]:
            lst = [i[key] for i in data_list]
            if key in list_smiles:
                batch[key] = lst
            else:   
                batch[key] = torch.stack(lst)
        
        return batch
class TransformerDataset_Rh(Dataset):
    """

    """
    def __init__(self, if_shuffle, data):
        # (L, B, C)
        self.data = data # list of dict of tensors
        self.shuffle = if_shuffle
        for feature_dict in data:
            for key in feature_dict:
                if key == 'ddG':
                    tmp = torch.tensor(feature_dict[key])     
                    feature_dict[key] = tmp
                else:
                    tmp = torch.tensor(feature_dict[key])     
                    feature_dict[key] = tmp.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            'reaction': data['reaction'],
            'Output': data['ddG']
        }
    
    def collate_fn(data_list):

        batch = {}
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        for key in data_list[0]:
            lst = [i[key] for i in data_list]
            if key in list_smiles:
                batch[key] = lst
            else:   
                batch[key] = torch.stack(lst)
        
        return batch