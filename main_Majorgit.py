

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
from model import ChemAHNet_Major, ChemAHNet_Major_MoIM, ChemAHNet_Major_RCIM, ChemAHNet_Major_MIM
from dataset import TransformerDataset
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
        # embedding_dim = self.args.embed_dim
        # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=True)
        self.model = ChemAHNet_Major(args, vocab_size, embedding_dim).to(self.device)
        # self.model = ChemAHNet_Major_MoIM(args, vocab_size, embedding_dim).to(self.device)
        
        self.model.embedding.weight = nn.Parameter(pretrained_embeddings)
        # self.model.embedding.weight.requires_grad = False
        # Initialize DistributedDataParallel
        # self.model = nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)     
    
        if not self.dataloader_tr is None:
            self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.args.lr, weight_decay=1e-5) 
            self.scheduler = WarmupLinearSchedule(self._optimizer, warmup_steps=5000, t_total=args.epochs*len(self.dataloader_tr))
        if args.checkpoint is not None:
            self.initialize_from_checkpoint()
            
        self.logger = None
        # if self.rank is 0:
        #     self.logger = SummaryWriter("log/"+args.name)
        #     self.logger.add_text('args', str(self.args), 0)
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
        }, args.save_path + 'best_Major_model_%d' % 250)

    def fit(self):
        t_total = time()
        total_loss = []
        print('start fitting')
        best_acc =0.0
        counter =0
        patience =80
        train_acc=[]
        train_loss = []
        valid_acc=[]
        valid_loss = []
        # test_result = trainer.test()
        for _ in range(self.args.epochs):
            epoch_loss,train_accuracy = self.train_epoch(self.epoch)
            self.epoch_loss = epoch_loss

            print('training loss %.4f' % epoch_loss)

            total_loss.append(epoch_loss)
            train_loss.append(epoch_loss)
            train_acc.append(train_accuracy)
            
            if self.epoch % 1 == 0:
                validation_loss , eval_accuracy  = self.evaluate()
                valid_loss.append(validation_loss)
                valid_acc.append(eval_accuracy)
                print('epoch %d eval_acc: %.4f' % (self.epoch, eval_accuracy))
                if eval_accuracy > best_acc:
                    self.save_model()
                    best_acc = eval_accuracy
                    counter =0
                elif eval_accuracy < best_acc:
                    counter += 1
                if counter >=patience:
                    print(f'Early stopping after {self.epoch} epochs due to no improvement in validation loss.')
                    break
            self.epoch += 1
        print('optimization finished ')
        print('Total tiem elapsed: {:.4f}s'.format(time() - t_total))

        test_result = trainer.test()
        test_acc = test_result['test_accuracy']
        test_acc_list = [test_acc] * len(train_acc)
        data = {
                'Epoch': list(range(len(train_acc))),
                'Train Accuracy': train_acc,
                'Validation Accuracy': valid_acc,
                'Test Accuracy': test_acc_list
                }
        df = pd.DataFrame(data)
        df.to_csv(args.save_path + 'ChemAHNet_Major_Training_Log.csv', index=False)



    def train_epoch(self, epoch_cnt):
        batch_losses = []
        running_corrects = 0
        total_samples = 0
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
                # batch_data[key] = batch_data[key].to(self.rank)
                    batch_gpu[key] = batch_data[key].to(self.device)
            Reactant = batch_gpu['Reactants'].long()       
            output = self.model(batch_gpu)
            criterion = nn.BCELoss().to(self.device)
            target = batch_data['Label'].unsqueeze(dim=1).float().to(self.device)
            # target = batch_data['target_val'].unsqueeze(dim=1).float().to(self.device)
            loss = criterion(output, target)
            if self.rank is 0:
                pbar.set_postfix(n=self.args.name, 
                                 b='{:.2f}'.format(loss))
            
            loss.backward()
            batch_losses.append(loss.item())
            self._optimizer.step()
            self.scheduler.step()
            
            if self.step % 10 == 0 and self.logger:
                self.logger.add_scalar('loss/total', loss.item(), self.step)


            predicted = (output >= 0.5).float()  
            correct_predictions = (predicted == target)  
            running_corrects += correct_predictions.sum().item()
            total_samples += target.size(0)
            self.step += 1
        train_accuracy = running_corrects / total_samples
        print('train acc', train_accuracy)
        if self.logger:
            self.logger.add_scalar('acc/train_acc', train_accuracy, self.epoch)
        epoch_loss = np.mean(np.array(batch_losses, dtype=float))
        
        return epoch_loss,train_accuracy
    
    def evaluate(self):
        running_corrects = 0
        total_samples = 0
        batch_losses = []
        all_outputs = []
        all_targets = []
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
                    # batch_data[key] = batch_data[key].to(self.rank)
                        batch_gpu[key] = batch_data[key].to(self.device)
                Reactant = batch_gpu['Reactants'].long()       
                output = self.model(batch_gpu)
                criterion = nn.BCELoss().to(self.device)
                target = batch_gpu['Label'].unsqueeze(dim=1).float().to(self.device)
                loss = criterion(output, target)
                batch_losses.append(loss.item())


                predicted = (output >= 0.5).float() 
                correct_predictions = (predicted == target)  

                running_corrects += correct_predictions.sum().item()
                total_samples += target.size(0)
            eval_accuracy = running_corrects / total_samples
            epoch_loss = np.mean(np.array(batch_losses, dtype=float))

            print('eval loss %.4f' % epoch_loss)
            print('eval_accuracy %.4f' % eval_accuracy)
            if self.logger:
                # self.logger.add_scalar('acc/accuracy', best_accuracy, self.epoch)
                self.logger.add_scalar('epoch_loss', epoch_loss, self.epoch)
            
            return epoch_loss,eval_accuracy

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
            model_path = os.path.join(args.save_path, 'best_Major_model_250')
            model_path = os.path.join("./Trained model/best_Major_model")
            # checkpoint = torch.load(model_path) #"gpu"
            checkpoint = torch.load(model_path, map_location=torch.device('cpu')) #"cpu"
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()
            pbar = tqdm(self.dataloader_te)
            for batch_data in iter(pbar):
                batch_gpu = {}
                for key in batch_data:
                    if key not in list_smiles:
                    # batch_data[key] = batch_data[key].to(self.rank)
                        batch_gpu[key] = batch_data[key].to(self.device)
                Reactant = batch_gpu['Reactants'].long()       
                output = self.model(batch_gpu)
                criterion = nn.BCELoss().to(self.device)
                target = batch_gpu['Label'].unsqueeze(dim=1).float().to(self.device)
                # all_targets.extend(target.cpu().numpy())
                loss = criterion(output, target)
                validation_loss += loss.item()
                predicted = (output >= 0.5).float()  
                correct_predictions = (predicted == target)  

                running_corrects += correct_predictions.sum().item()
                total_samples += target.size(0)

                all_outputs.extend(output.detach().cpu().numpy())  
                all_targets.extend(target.detach().cpu().numpy())  

                        
            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)
            for threshold in np.arange(0.0, 1.01, 0.01):
                predicted = (all_outputs >= threshold).astype(int)  
                accuracy = accuracy_score(all_targets, predicted)  
    
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            print(f"Validation Loss: {validation_loss / len(self.dataloader_te):.4f}")
            test_accuracy = running_corrects / total_samples
            print('test_accuracy %.4f' % test_accuracy)
            print('best acc %.4f' % best_accuracy)
            print('best_threshold %.4f' % best_threshold)
            if self.logger:
                self.logger.add_scalar('acc/accuracy', best_accuracy, self.epoch)                 
        test_result['Validation Loss'] = validation_loss / len(self.dataloader_te)

        test_result['test_accuracy'] = test_accuracy
        test_result['best acc'] = best_accuracy
        return test_result
 


def load_data(args, name):
    file = open('data/' + name + '.pickle', 'rb')
    full_data = pickle.load(file)
    file.close()
    full_dataset = TransformerDataset(args.shuffle, full_data)

    data_loader = DataLoader(full_dataset,
                             batch_size=args.batch_size,
                             shuffle=(name == 'example_train_data'),
                             num_workers=args.num_workers, collate_fn = TransformerDataset.collate_fn)
    return data_loader



                
                
if __name__ == '__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info') 
    
    args = parser.parse_args()
    seed = args.seed
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
    # if args.train:
    train_dataloader = load_data(args, 'example_train_data')
    valid_dataloader = load_data(args, 'example_eval_data')
# if args.test:
    test_dataloader = load_data(args, 'example_test_data')

    trainer = Trainer(train_dataloader, test_dataloader, valid_dataloader, args)
    
    if args.train:
        trainer.fit()
    elif args.test:    
        trainer.test()