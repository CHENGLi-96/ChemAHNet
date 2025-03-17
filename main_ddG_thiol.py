

# coding: utf-8

from time import time
import numpy as np
import math
import os
import sys
import json
from tqdm import tqdm, trange
import pdb
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from model import ChemAHNet_ddG
from dataset import TransformerDataset_Thiol,TransformerDataset
from config import parser
import os
import pickle
import torch.distributed as dist
from concurrent.futures import ProcessPoolExecutor
from rdkit import RDLogger
from optimization import WarmupLinearSchedule
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split

class Trainer(object):
    def __init__(self, dataloader_tr, dataloader_te, dataloader_val, args):
        self.dataloader_tr = dataloader_tr
        self.dataloader_te = dataloader_te
        self.dataloader_val = dataloader_val
        self.args = args
        self.rank = args.local_rank


        self.ntoken = 100
        self.step = 0
        self.eval_step = 0
        self.epoch = 0
        self.epoch_loss = 100
        self.device =args.device
        model_path = "./hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots/761d6a18cf99db371e0b43baf3e2d21b3e865a20"
        chem_model = AutoModel.from_pretrained(model_path)
        pretrained_embeddings = chem_model.embeddings.word_embeddings.weight.detach().clone().to(self.device)
        vocab_size = pretrained_embeddings.shape[0]
        embedding_dim = pretrained_embeddings.shape[1]  
        self.model = ChemAHNet_ddG(args, vocab_size, embedding_dim).to(self.device)
        self.model.embedding.weight = nn.Parameter(pretrained_embeddings)   
    
        if not self.dataloader_tr is None:
            self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.args.lr, weight_decay=1e-5) 
            self.scheduler = WarmupLinearSchedule(self._optimizer, warmup_steps=500, t_total=args.epochs*len(self.dataloader_tr))
        if args.checkpoint is not None:
            self.initialize_from_checkpoint()
            
        self.logger = None
        if self.rank == 0:
            log_dir = os.path.join("log", args.name)
            print(f"Log directory is set to: {log_dir}")
            if os.path.exists(log_dir):
                if os.path.isfile(log_dir):
                    print(f"Removing conflicting file at {log_dir}")
                    os.remove(log_dir)
                elif os.path.isdir(log_dir):
                    print(f"Directory {log_dir} already exists")
                else:
                    print(f"Unknown type at {log_dir}, removing it.")
                    os.remove(log_dir) 

            if not os.path.exists(log_dir): 
                print(f"Creating directory {log_dir}")
                try:
                    os.makedirs(log_dir)
                    print(f"Successfully created the directory {log_dir}")
                except Exception as e:
                    print(f"Failed to create directory {log_dir}: {e}")
                    raise

            self.logger = SummaryWriter(log_dir)
            self.logger.add_text('args', str(self.args), 0)

        
    def initialize_from_checkpoint(self):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        # map_location = 'cpu'  
        state_dict = {}
        for checkpoint in self.args.checkpoint:
            checkpoint = torch.load(os.path.join(self.args.save_path, checkpoint), map_location=map_location)
            for key in checkpoint['model_state_dict']:
                if key in state_dict:
                    state_dict[key] += checkpoint['model_state_dict'][key]
                else:
                    state_dict[key] = checkpoint['model_state_dict'][key]
        for key in state_dict:
            state_dict[key] = state_dict[key]/len(self.args.checkpoint)
        self.model.module.load_state_dict(state_dict)    
        if self.dataloader_tr is not None:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        print('initialized!')

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, args.save_path + 'Trans_model_%d' % 250)

    def fit(self):
        t_total = time()
        total_loss = []
        print('start fitting')
        best_acc =0.0
        counter =0
        patience =80
        best_loss = 1000000
        best_r2 = -100000
        tr_mae =[]
        tr_r2 = []
        ev_mae =[]
        ev_r2 = []
        for _ in range(self.args.epochs):
            epoch_loss,train_mae,train_r2 = self.train_epoch(self.epoch)
            self.epoch_loss = epoch_loss

            print('training loss %.4f' % epoch_loss)
            print('train_r2 %.4f' % train_r2)
            total_loss.append(epoch_loss)
            tr_mae.append(train_mae)
            tr_r2.append(train_r2)
            
            if self.epoch % 1 == 0:
                validation_loss,eval_mae, eval_r2  = self.evaluate()
                ev_mae.append(eval_mae)
                ev_r2.append(eval_r2)                
                if eval_r2 > best_r2:
                    self.save_model()
                    best_r2 = eval_r2
                    counter =0
                elif best_r2 > eval_r2:
                    counter += 1
                if counter >=patience:
                    print(f'Early stopping after {self.epoch} epochs due to no improvement in validation loss.')
                    break
            self.epoch += 1
        print('optimization finished ')
        print('Total tiem elapsed: {:.4f}s'.format(time() - t_total))
        test_result = trainer.test()

 
        



    def train_epoch(self, epoch_cnt):
        batch_losses = []
        running_corrects = 0
        total_samples = 0
        train_outputs = []
        train_targets = []
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        self.model.train()
        torch.cuda.empty_cache()  
        if self.rank is 0:
            pbar =  tqdm(self.dataloader_tr)
        else:
            pbar = self.dataloader_tr
        for batch_data in pbar:
            batch_gpu = {}
            self._optimizer.zero_grad()
            for key in batch_data:
                if key not in list_smiles:
                    batch_gpu[key] = batch_data[key].to(self.device)     
            output = self.model(batch_gpu)
            criterion = nn.MSELoss().to(self.device)
            target = batch_data['Output'].unsqueeze(dim=1).float().to(self.device)
            loss = criterion(output, target)
            output= output*( 3.134624572 + 0.419377753) - 0.419377753
            target= target*( 3.134624572 + 0.419377753) - 0.419377753 
            train_targets.extend(target.detach().cpu().numpy())
            train_outputs.extend(output.detach().cpu().numpy())
            if self.rank is 0:
                pbar.set_postfix(n=self.args.name, 
                                 b='{:.2f}'.format(loss))
            
            loss.backward()
            batch_losses.append(loss.item())
            self._optimizer.step()
            self.scheduler.step()
            
            if self.step % 10 == 0 and self.logger:
                self.logger.add_scalar('loss/total', loss.item(), self.step)

            self.step += 1

        epoch_loss = np.mean(np.array(batch_losses, dtype=float))
        train_outputs = np.array(train_outputs)
        train_targets = np.array(train_targets)
        train_mae = mean_absolute_error(train_targets, train_outputs)
        train_r2 = r2_score(train_targets, train_outputs)
        return epoch_loss ,train_mae,train_r2
    
    def evaluate(self):
        running_corrects = 0
        total_samples = 0
        eval_outputs = []
        eval_targets = []
        best_threshold = 0.0
        best_accuracy = 0.0
        validation_loss = 0.0
        best_loss = 10000000
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(self.dataloader_val)

            for batch_data in iter(pbar):
                batch_gpu = {}
                for key in batch_data:
                    if key not in list_smiles:
                        batch_gpu[key] = batch_data[key].to(self.device)      
                output = self.model(batch_gpu)

                criterion = nn.MSELoss().to(self.device)
                target = batch_data['Output'].unsqueeze(dim=1).float().to(self.device)
                loss = criterion(output, target)
                validation_loss += loss.item()
                output= output*( 3.134624572 + 0.419377753) - 0.419377753
                target= target*( 3.134624572 + 0.419377753) - 0.419377753
                eval_targets.extend(target.detach().cpu().numpy())
                eval_outputs.extend(output.detach().cpu().numpy())
            print('eval loss %.4f' % validation_loss)
            if self.logger:
                self.logger.add_scalar('validation_loss', validation_loss, self.epoch)
            eval_outputs = np.array(eval_outputs)
            eval_targets = np.array(eval_targets)
            eval_mae = mean_absolute_error(eval_targets, eval_outputs)
            eval_r2 = r2_score(eval_targets, eval_outputs)
            print('eval_r2 %.4f' % eval_r2)
            return validation_loss ,eval_mae,eval_r2

    def test(self):
        
        running_corrects = 0
        total_samples = 0
        test_result = {}
        pred  = []
        incorrect_predictions = [] 
        all_outputs = []
        all_targets = []
        best_threshold = 0.0
        best_accuracy = 0.0
        validation_loss = 0.0
        all_outputs = []
        all_targets = []
        list_smiles = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','reaction_SMILES']
        test_smile = ['Reactant', 'Solvent', 'Catalyst', 'Product SMILES','Label','reaction_SMILES']
        with torch.no_grad():
            model_path = os.path.join(args.save_path, 'Trans_model_250')
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()
            pbar = tqdm(self.dataloader_te)
            for batch_data in iter(pbar):
                batch_gpu = {}
                for key in batch_data:
                    if key not in list_smiles:
                        batch_gpu[key] = batch_data[key].to(self.device)    
                output = self.model(batch_gpu)
                criterion = nn.MSELoss().to(self.device)
                target = batch_data['Output'].unsqueeze(dim=1).float().to(self.device)
                loss = criterion(output, target)
                validation_loss += loss.item()
                output= output*( 3.134624572 + 0.419377753) - 0.419377753
                target= target*( 3.134624572 + 0.419377753) - 0.419377753 
                all_outputs.extend(output.detach().cpu().numpy())  
                all_targets.extend(target.detach().cpu().numpy())  

                        
            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            mae = mean_absolute_error(all_targets, all_outputs)
            r2 = r2_score(all_targets, all_outputs)
            print(f"Validation Loss: {validation_loss / len(self.dataloader_te):.4f}")
               
        test_result['Validation Loss'] = validation_loss / len(self.dataloader_te)
        test_result['mae'] = mae
        test_result['r2'] = r2

        plt.scatter(all_targets, all_outputs, color='blue', alpha=0.6, label='数据点')


        all_targets_1d = np.array(all_targets).flatten()  
        all_outputs_1d = np.array(all_outputs).flatten()  
        sns.set(style='darkgrid')
        fig = plt.figure(figsize=(11,11),facecolor='white',   
                edgecolor='black')
        plt.scatter(all_targets_1d,all_outputs_1d,s=250, c='royalblue', label="samples",alpha=0.6,edgecolors='navy')
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], color='red', linestyle='--', label='理想拟合')

        plt.xlim(-1,4)
        plt.ylim(-1,4)
        x_major_locator=MultipleLocator(1)
        y_major_locator=MultipleLocator(1)
        ax=plt.gca()

        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlabel("Observed $\Delta$$\Delta$G(kcal/mol)",fontsize=36)
        plt.ylabel("Predicted $\Delta$$\Delta$G(kcal/mol)",fontsize=36)

        plt.tick_params(labelsize=30)
        plt.text(-0.8,3.1,'RMSE = %.3f'%(mean_squared_error(all_targets_1d,all_outputs_1d))**(0.5),fontsize=30)
        plt.text(-0.8,3.45,'R$^2$ = %.3f'%r2_score(all_targets_1d,all_outputs_1d),fontsize=30)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(args.save_path + 'regression_fit_NSTransformation.png')
        NS_data = {
            'targets': all_targets_1d,  
            'outputs': all_outputs_1d  
        }
        df = pd.DataFrame(NS_data)

  
        df.to_csv(args.save_path + 'targets_and_outputs_NSTransformation.csv', index=False)  

        return test_result
 


def load_data(args, name):
    file = open('data/' + name + '.pickle', 'rb')
    full_data = pickle.load(file)
    file.close()
    test_size = 475 / 1075
    train_data, test_data = train_test_split(full_data, test_size=test_size, random_state=42)
    train_dataset = TransformerDataset_Thiol(args.shuffle, train_data)
    val_dataset = TransformerDataset_Thiol(args.shuffle, test_data)
    test_dataset = TransformerDataset_Thiol(args.shuffle, test_data)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers, 
                              collate_fn=TransformerDataset_Thiol.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers, 
                            collate_fn=TransformerDataset_Thiol.collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers, 
                             collate_fn=TransformerDataset_Thiol.collate_fn)
    
    return train_loader, val_loader, test_loader

def load_pdata(args, name):
    file = open('data/' + name + '.pickle', 'rb')
    full_data = pickle.load(file)
    file.close()
    full_dataset = TransformerDataset_Thiol(args.shuffle, full_data)

    data_loader = DataLoader(full_dataset,
                             batch_size=args.batch_size,
                             shuffle=(name == 'train_Transformation_data'),
                             num_workers=args.num_workers, collate_fn = TransformerDataset_Thiol.collate_fn)
    return data_loader

                
                
if __name__ == '__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info') 
    
    args = parser.parse_args()
    seed = args.seed + args.local_rank
    np.random.seed(seed)
    torch.manual_seed(seed) 
    # torch.cuda.manual_seed(seed)
    # torch.cuda.set_device(args.local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        # backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    args.device = device
    args.save_path = args.save_path + '/' + args.name + '/'
    
    # if not os.path.exists(args.save_path):
    #     os.mkdir(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)  


    print(args)
    # dist.init_process_group("nccl", rank=args.local_rank, world_size=args.world_size)
    # dist.init_process_group(backend, rank=args.local_rank, world_size=args.world_size)
    # pdb.set_trace()
    valid_dataloader = None
    train_dataloader = None
    test_dataloader = None
    if args.train:
        # train_dataloader, test_dataloader, valid_dataloader = load_data(args, 'output_data')
        train_dataloader = load_pdata(args, 'train_Transformation_data')
        valid_dataloader = load_pdata(args, 'test_Transformation_data')
    # # if args.test:
        test_dataloader = load_pdata(args, 'test_Transformation_data')

    # trainer = Trainer(train_dataloader, test_dataloader, valid_dataloader, args)

    trainer = Trainer(train_dataloader, test_dataloader, valid_dataloader, args)
    
    if args.train:
        args.lr = 1e-3
        trainer.fit()
    elif args.test:    
        trainer.test()