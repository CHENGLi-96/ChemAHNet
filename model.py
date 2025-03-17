from torch import nn
import torch
import torch.nn.functional as F
import math
import pdb
import os

from layer import *
from functools import partial
from transformers import AutoTokenizer
import numpy as np
from math import floor



# class MoleculeEncoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.transform2D_matrix = nn.Parameter(torch.randn(2048, 8*self.dim)).to(self.device)
        


#     def forward(self, element, bond, aroma, charge, mask, segment, reactant=None):

#         embedding = self.atom_encoder(element, bond, aroma, charge, segment, reactant)

#         encoder_output = self.transformer_encoder(embedding, src_key_padding_mask=mask)
#         return encoder_output

model_path = "./hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots/761d6a18cf99db371e0b43baf3e2d21b3e865a20"


BN = partial(nn.BatchNorm1d, eps=5e-3,momentum=0.1)

##############################################################################################
class LinearBnRelu(nn.Module):
	def __init__(self, in_dim, out_dim, is_bn=True):
		super().__init__()
		self.is_bn = is_bn
		self.linear = nn.Linear(in_dim, out_dim, bias=not is_bn)
		self.bn=BN(out_dim)

	def forward(self, x):
		x = self.linear(x)
		if self.is_bn:
			x=self.bn(x)
		x = F.relu(x,inplace=True)
		return x

class Conv1dBnRelu(nn.Module):
	def __init__(self, in_channel,out_channel,kernel_size,stride=1,padding=0, is_bn=True):
		super().__init__()
		self.is_bn = is_bn
		self.conv=nn.Conv1d(
			in_channel, out_channel, kernel_size=kernel_size,
			stride=stride, padding=padding, bias=not is_bn
		)
		self.bn=BN(out_channel)

	def forward(self, x):
		x=self.conv(x)
		if self.is_bn:
			x=self.bn(x)
		x = F.relu(x,inplace=True)
		return x

##############################################################################################


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, max_len=128):
		super(PositionalEncoding, self).__init__()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[ :,:x.size(1)]
		return x

class ChemAHNet_Major(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.args.output_dim, self.args.output_dim)	
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)

        
        # 定义 Softmax 激活函数
		self.softmax = nn.Softmax(dim=-1)  # 在最后一个维度上应用 softmax
		self.attention_depth_inter=2
		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)


		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
		# self.SENst = SEAttention(channel=3,reduction=8).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)
		# self.output = nn.Sequential(nn.Linear(32, 1))  # 二分类输出
		# init.xavier_uniform_(self.output[0].weight)
		# init.zeros_(self.output[0].bias)
		# self.fc1 = nn.Linear(32, 32, bias=True)
		# self.fc2 = nn.Linear(32, 32, bias=True)
		self.attention_weights = nn.Parameter(torch.randn(self.args.output_dim))
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		# self.layer_norm = nn.LayerNorm(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		# self.resblock4 = ResidualBlock(2048, 128,0.1)
		# self.resblock5 = ResidualBlock(128, 64,0.1)
		# self.resblock6 = ResidualBlock(64, 32,0.1)
		self.attention = nn.MultiheadAttention(self.args.output_dim, 8)
		self.alpha_predictor = nn.Sequential(
            nn.Linear(self.args.output_dim, self.args.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.args.output_dim // 2, 1),
            nn.Sigmoid()
        )
		self.output_layer = nn.Linear(self.args.output_dim, self.args.output_dim)
		        # 注意力融合模块
		# self.attention_fc = nn.Sequential(
        #     nn.Linear(self.args.output_dim * 3, 128),  # 输入为3个卷积特征flatten后的总维度
        #     nn.ReLU(),
        #     nn.Linear(128, 3),  # 为每种卷积特征分配权重
        # )
		# self.univariate_layers = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.args.output_dim*self.args.max_length, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # ) for _ in range(3)])
		# self.output_layer = nn.Linear(1 * 3, 1)
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		# self.fc_gate = nn.Linear(self.args.output_dim*self.args.max_length, self.args.output_dim*self.args.max_length)  
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
		# Solvent_SMILES = (Solvent_SMILES != 1).long()
		# Reactant_SMILES = (Reactant_SMILES != 1).long()
		# B, L  = Reactant.shape
		# print(smiles,B,L)

				Reactant_x = self.embedding(Reactant)
				batch_dim = Reactant_x.shape[0]
				# Reactant_x = self.reduce_dim(Reactant_x)
				# Reactant_x = self.pe(Reactant_x)
		# Solvent_x = self.embedding(Solvent_SMILES)
		# Reactant_x = self.embedding(Reactant_SMILES)

		# Reactant_x = self.reduce_dim(Reactant_x)
		# 尝试卷积
				Reactant_a = Reactant_x.permute(0,2,1).float()
				Reactant_x = self.conv_embedding(Reactant_a)
				Reactant_x = Reactant_x.permute(0,2,1).contiguous()
		
				Reactant_x = self.pe(Reactant_x)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0

				Reactant_x5 = self.conv5_embedding(Reactant_a)
				Reactant_x5 = Reactant_x5.permute(0,2,1).contiguous()
		
				Reactant_x5 = self.pe(Reactant_x5)
				Reactant_z5 = self.tx_encoder(
					x=Reactant_x5,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z5[Reactant_mask == 0] = 0

				Reactant_x7 = self.conv7_embedding(Reactant_a)
				Reactant_x7 = Reactant_x7.permute(0,2,1).contiguous()
				Reactant_x7 = self.pe(Reactant_x7)
				Reactant_z7 = self.tx_encoder(
					x=Reactant_x7,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z7[Reactant_mask == 0] = 0


		Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		# Reactant_z = F.relu(Reactant_z + Reactant_z5 + Reactant_z7)
		Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim*3).squeeze(-1)
		Reactant_z = self.batch_norm(Reactant_z)
		Reactant_z = self.resblock1(Reactant_z)
		Reactant_z = self.resblock2(Reactant_z)
		Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)
		bind = torch.sigmoid(bind)

		return bind
	def get_pred_list(self,tlist):
		predlist = []
		for text in tlist:
			Reactant = self.tokenfun(text)
			Reactant_mask = (Reactant != 1).bool()
		# B, L  = Reactant.shape
		# print(smiles,B,L)

			# Reactant = batch[smiles].long()
			# Reactant_mask = (Reactant != 1).bool()
	# Solvent_SMILES = (Solvent_SMILES != 1).long()
	# Reactant_SMILES = (Reactant_SMILES != 1).long()
	# B, L  = Reactant.shape
	# print(smiles,B,L)

			Reactant_x = self.embedding(Reactant)
			# batch_dim = Reactant_x.shape[0]
	# Solvent_x = self.embedding(Solvent_SMILES)
	# Reactant_x = self.embedding(Reactant_SMILES)

	# Reactant_x = self.reduce_dim(Reactant_x)
	# 尝试卷积
			Reactant_a = Reactant_x.permute(0,2,1).float()
			Reactant_x = self.conv_embedding(Reactant_a)
			Reactant_x = Reactant_x.permute(0,2,1).contiguous()
			# Reactant_x = self.conv_embedding[1](Reactant_x)
	
			Reactant_x = self.pe(Reactant_x)
			Reactant_z = self.tx_encoder(
				x=Reactant_x,
				# src_key_padding_mask=smiles_token_mask==0
				src_key_padding_mask=None,
			)
		# 手动恢复填充位置的输出
			Reactant_z[Reactant_mask == 0] = 0

			Reactant_x5 = self.conv5_embedding(Reactant_a)
			Reactant_x5 = Reactant_x5.permute(0,2,1).contiguous()
			# Reactant_x5 = self.conv5_embedding[1](Reactant_x5)
			Reactant_x5 = self.pe(Reactant_x5)
			Reactant_z5 = self.tx_encoder(
				x=Reactant_x5,
				# src_key_padding_mask=smiles_token_mask==0
				src_key_padding_mask=None,
			)
		# 手动恢复填充位置的输出
			Reactant_z5[Reactant_mask == 0] = 0

			Reactant_x7 = self.conv7_embedding(Reactant_a)
			Reactant_x7 = Reactant_x7.permute(0,2,1).contiguous()
			# Reactant_x7 = self.conv5_embedding[1](Reactant_x7)
			Reactant_x7 = self.pe(Reactant_x7)
			Reactant_z7 = self.tx_encoder(
				x=Reactant_x7,
				# src_key_padding_mask=smiles_token_mask==0
				src_key_padding_mask=None,
			)
			Reactant_z7[Reactant_mask == 0] = 0
			Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		# print(Reactant_z.size())
			Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.max_length*3).squeeze(-1)
			# terms = [layer(Reactant_z) for layer in self.univariate_layers]
			# combined = torch.cat(terms, dim=-1)
			# output = self.output_layer(combined)
			Reactant_z = self.batch_norm(Reactant_z)
			Reactant_z = self.resblock1(Reactant_z)
			Reactant_z = self.resblock2(Reactant_z)
			Reactant_z = self.resblock3(Reactant_z)
			bind = self.bind(Reactant_z)
			bind = torch.sigmoid(bind)
			output = bind.detach().cpu().numpy().squeeze()
			if output.ndim == 0:
				output = np.array([output])

        # 为每个输出生成 [1 - output, output] 概率分布
			for out in output:
				proba = [1 - out, out]
				predlist.append(proba)

		# --------------------------
		return np.array(predlist)
class ChemAHNet_Major_MoIM(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.args.output_dim, self.args.output_dim)	
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)

        
        # 定义 Softmax 激活函数
		self.softmax = nn.Softmax(dim=-1)  # 在最后一个维度上应用 softmax
		self.attention_depth_inter=2
		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)


		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
		# self.SENst = SEAttention(channel=3,reduction=8).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)

		self.attention_weights = nn.Parameter(torch.randn(self.args.output_dim))
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		# self.layer_norm = nn.LayerNorm(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)

		self.attention = nn.MultiheadAttention(self.args.output_dim, 8)
		self.alpha_predictor = nn.Sequential(
            nn.Linear(self.args.output_dim, self.args.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.args.output_dim // 2, 1),
            nn.Sigmoid()
        )
		self.output_layer = nn.Linear(self.args.output_dim, self.args.output_dim)

		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		# self.fc_gate = nn.Linear(self.args.output_dim*self.args.max_length, self.args.output_dim*self.args.max_length)  
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
		# Solvent_SMILES = (Solvent_SMILES != 1).long()
		# Reactant_SMILES = (Reactant_SMILES != 1).long()
		# B, L  = Reactant.shape
		# print(smiles,B,L)

				Reactant_x = self.embedding(Reactant)
				batch_dim = Reactant_x.shape[0]
				Reactant_a = Reactant_x.permute(0,2,1).float()
				Reactant_x = self.conv_embedding(Reactant_a)
				Reactant_x = Reactant_x.permute(0,2,1).contiguous()
		
				Reactant_x = self.pe(Reactant_x)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0

		# Reactant_z = F.relu(Reactant_z + Reactant_z5 + Reactant_z7)
		Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim).squeeze(-1)
		Reactant_z = self.batch_norm(Reactant_z)
		Reactant_z = self.resblock1(Reactant_z)
		Reactant_z = self.resblock2(Reactant_z)
		Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)
		bind = torch.sigmoid(bind)

		return bind
class ChemAHNet_Major_RCIM(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.args.output_dim, self.args.output_dim)	
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)
		self.reduce_dim1 = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.reduce_dim2 = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.reduce_dim3 = nn.Linear(self.embedding_dim, self.args.output_dim)
        # 定义 Softmax 激活函数
		self.softmax = nn.Softmax(dim=-1)  # 在最后一个维度上应用 softmax
		self.attention_depth_inter=2
		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)


		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
		# self.SENst = SEAttention(channel=3,reduction=8).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)

		self.attention_weights = nn.Parameter(torch.randn(self.args.output_dim))
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		# self.layer_norm = nn.LayerNorm(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		self.attention = nn.MultiheadAttention(self.args.output_dim, 8)
		self.alpha_predictor = nn.Sequential(
            nn.Linear(self.args.output_dim, self.args.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.args.output_dim // 2, 1),
            nn.Sigmoid()
        )
		self.output_layer = nn.Linear(self.args.output_dim, self.args.output_dim)
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		# self.fc_gate = nn.Linear(self.args.output_dim*self.args.max_length, self.args.output_dim*self.args.max_length)  
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()

				Reactant_a = self.embedding(Reactant)
				Reactant_x = self.reduce_dim1(Reactant_a)
				Reactant_x = self.pe(Reactant_x)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0
				Reactant_x5 = self.reduce_dim2(Reactant_a)
				Reactant_x5 = self.pe(Reactant_x5)
				Reactant_z5 = self.tx_encoder(
					x=Reactant_x5,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z5[Reactant_mask == 0] = 0


				Reactant_x7 = self.reduce_dim3(Reactant_a)
				Reactant_x7 = self.pe(Reactant_x7)
				Reactant_z7 = self.tx_encoder(
					x=Reactant_x7,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z7[Reactant_mask == 0] = 0


		Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		# Reactant_z = F.relu(Reactant_z + Reactant_z5 + Reactant_z7)
		Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim*3).squeeze(-1)
		Reactant_z = self.batch_norm(Reactant_z)
		Reactant_z = self.resblock1(Reactant_z)
		Reactant_z = self.resblock2(Reactant_z)
		Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)
		bind = torch.sigmoid(bind)

		return bind
class ChemAHNet_Major_MIM(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)
        # 定义 Softmax 激活函数

		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.output = nn.Sequential(nn.Linear(32, 1))  # 二分类输出
		init.xavier_uniform_(self.output[0].weight)
		init.zeros_(self.output[0].bias)
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		self.self_gate = SelfGate(self.args.output_dim)
		self.reduce_dim2 = nn.Linear(self.args.output_dim, 32)
		# self.fusion_layer = GatedFeatureFusion(2048)
		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim*4,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
	

		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double','reaction']
		# list_smiles = ['reaction','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
				Reactant_x = self.embedding(Reactant)

				Reactant_a = Reactant_x.permute(0,2,1).float()
				Reactant_x = self.conv_embedding(Reactant_a)
				Reactant_xc = Reactant_x.permute(0,2,1).contiguous()
				# Reactant_xc = self.self_gate(Reactant_xc)
				Reactant_x = self.pe(Reactant_xc)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0
				Reactant_x5 = self.conv5_embedding(Reactant_a)
				Reactant_x5c = Reactant_x5.permute(0,2,1).contiguous()
				# Reactant_x5c = self.self_gate(Reactant_x5c)
				Reactant_x5 = self.pe(Reactant_x5c)
				Reactant_z5 = self.tx_encoder(
					x=Reactant_x5,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z5[Reactant_mask == 0] = 0


				Reactant_x7 = self.conv7_embedding(Reactant_a)
				Reactant_x7c = Reactant_x7.permute(0,2,1).contiguous()
				# Reactant_x7c = self.self_gate(Reactant_x7c)
				Reactant_x7 = self.pe(Reactant_x7c)
				Reactant_z7 = self.tx_encoder(
					x=Reactant_x7,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z7[Reactant_mask == 0] = 0


		# Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		# Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		Reactant_z = Reactant_z + Reactant_z5 + Reactant_z7
		Reactant_z = Reactant_z.permute(0, 2, 1)
		Reactant_z = torch.sum(Reactant_z, dim=2)
		Reactant_z = self.reduce_dim2(Reactant_z)
		# Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim*3).squeeze(-1)
		# Reactant_z = self.batch_norm(Reactant_z)
		# Reactant_z = self.resblock1(Reactant_z)
		# Reactant_z = self.resblock2(Reactant_z)
		# Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)
		bind = torch.sigmoid(bind)
		return bind
class ChemAHNet_ddG(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)
		# self.feat_final_layer = nn.Linear(self.args.output_dim, 1).to(self.args.device)
		# self.batch_norm = nn.BatchNorm1d(1).to(self.args.device)
		        # 定义全连接层
		# self.fc_1 = nn.Linear(in_features=356, out_features=128)  # 第一层：输入128，输出128
		# self.fc_2 = nn.Linear(in_features=128, out_features=128)  # 第二层：输入128，输出128
		# self.fc_3 = nn.Linear(in_features=128, out_features=1)    # 第三层：输入128，输出1
		#         # Sigmoid 激活函数用于输出层
		# self.sigmoid = nn.Sigmoid()
		# self.fc_5 = nn.Linear(347, 32)
		# self.feat_number=32
		# self.super_feat_attention_dense = nn.Linear(self.feat_number, self.feat_number)
		# self.x_attention_dense = nn.Linear(356, 356)
		# self.inter_len=8
		# self.end_attention=1
		# self.conv2d_3 = nn.Conv2d(1, 1, kernel_size=(16, 16))
		# self.conv2d_4 = nn.Conv2d(1, 1, kernel_size=(8, 8))
		# self.conv2d_5 = nn.Conv2d(1, 1, kernel_size=(self.inter_len, self.inter_len))
		# self.layer4 = nn.Linear(self.args.output_dim*self.args.max_length, 2048).to(self.args.device)
		# self.layer5 = nn.Linear(2048, 2048).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)
        # 定义 Softmax 激活函数

		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		# self.conv = nn.Conv1d(self.embedding_dim, self.args.num_filter_maps, kernel_size=self.args.kernel_size,
        #                       padding=int(floor(self.args.kernel_size / 2)))
		# self.convs = nn.ModuleList([ 
		# 	nn.Conv1d(k*self.args.num_filter_maps, self.args.num_filter_maps, self.args.kernel_size, padding=int(floor(self.args.kernel_size / 2))) for k in np.arange(1,6)])
		# nn.init.xavier_uniform_(self.conv.weight)
		# for conv in self.convs:
		# 	nn.init.xavier_uniform_(conv.weight)

		# self.mlp = nn.Sequential(
		# 	nn.Linear(6, 200),
		# 	nn.ReLU(),
		# 	nn.Linear(200, 6))
		# self.embed_drop = nn.Dropout(p=self.args.dropout)
		# self.fc_out = nn.Linear(self.args.num_filter_maps, 1)
		# self.layer1 = nn.Linear(256, 128)
		# self.layer2 = nn.Linear(128, 64)
		# self.layer3 = nn.Linear(64, 32)
		self.output = nn.Sequential(nn.Linear(32, 1))  # 二分类输出
		init.xavier_uniform_(self.output[0].weight)
		init.zeros_(self.output[0].bias)
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		self.self_gate = SelfGate(self.args.output_dim)
		# self.reduce_dim2 = nn.Linear(self.args.output_dim, 32)
		# self.fusion_layer = GatedFeatureFusion(2048)
		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim*4,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
	

		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double','reaction']
		# list_smiles = ['reaction','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='reaction':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
		# Solvent_SMILES = (Solvent_SMILES != 1).long()
		# Reactant_SMILES = (Reactant_SMILES != 1).long()
		# B, L  = Reactant.shape
		# print(smiles,B,L)

				Reactant_x = self.embedding(Reactant)
				# Reactant_x = self.reduce_dim(Reactant_x)
				# Reactant_x = self.pe(Reactant_x)
		# Solvent_x = self.embedding(Solvent_SMILES)
		# Reactant_x = self.embedding(Reactant_SMILES)

		# Reactant_x = self.reduce_dim(Reactant_x)
		# 尝试卷积
				Reactant_a = Reactant_x.permute(0,2,1).float()
				Reactant_x = self.conv_embedding(Reactant_a)
				Reactant_xc = Reactant_x.permute(0,2,1).contiguous()
				# Reactant_xc = self.self_gate(Reactant_xc)
				Reactant_x = self.pe(Reactant_xc)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0
				Reactant_x5 = self.conv5_embedding(Reactant_a)
				Reactant_x5c = Reactant_x5.permute(0,2,1).contiguous()
				# Reactant_x5c = self.self_gate(Reactant_x5c)
				Reactant_x5 = self.pe(Reactant_x5c)
				Reactant_z5 = self.tx_encoder(
					x=Reactant_x5,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z5[Reactant_mask == 0] = 0


				Reactant_x7 = self.conv7_embedding(Reactant_a)
				Reactant_x7c = Reactant_x7.permute(0,2,1).contiguous()
				# Reactant_x7c = self.self_gate(Reactant_x7c)
				Reactant_x7 = self.pe(Reactant_x7c)
				Reactant_z7 = self.tx_encoder(
					x=Reactant_x7,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z7[Reactant_mask == 0] = 0


		Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim*3).squeeze(-1)
		Reactant_z = self.batch_norm(Reactant_z)
		Reactant_z = self.resblock1(Reactant_z)
		Reactant_z = self.resblock2(Reactant_z)
		Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)

		# bind = torch.sigmoid(bind)
		# for smiles in list_smiles:
		# if smiles =='double':
		# 	x_smiles = batch[smiles].squeeze(dim=1).to(self.args.device, dtype=torch.float32)
		# 	# print(x.size())
		# 	# x = self.fusion_layer(Reactant_z, x)
		# 	# x = torch.cat([Reactant_z, x], axis=1)
		# 	x = self.resblock4(x_smiles)
		# 	x = self.resblock5(x)
		# 	x = self.resblock6(x)
		# 	# x = torch.relu(self.layer1(x))
		# 	# x = torch.relu(self.layer2(x))
		# 	# x = torch.relu(self.layer3(x))
		# 	x = self.output(x)
		# 	x = self.sigmoid(x)  # 输出为概率
		# # x = 0.6*bind + 0.4*x
		return bind
class ChemAHNet_ddG_MoIM(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)
        # 定义 Softmax 激活函数

		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		# self.conv = nn.Conv1d(self.embedding_dim, self.args.num_filter_maps, kernel_size=self.args.kernel_size,
        #                       padding=int(floor(self.args.kernel_size / 2)))
		# self.convs = nn.ModuleList([ 
		# 	nn.Conv1d(k*self.args.num_filter_maps, self.args.num_filter_maps, self.args.kernel_size, padding=int(floor(self.args.kernel_size / 2))) for k in np.arange(1,6)])
		# nn.init.xavier_uniform_(self.conv.weight)
		# for conv in self.convs:
		# 	nn.init.xavier_uniform_(conv.weight)

		# self.mlp = nn.Sequential(
		# 	nn.Linear(6, 200),
		# 	nn.ReLU(),
		# 	nn.Linear(200, 6))
		# self.embed_drop = nn.Dropout(p=self.args.dropout)
		# self.fc_out = nn.Linear(self.args.num_filter_maps, 1)
		# self.layer1 = nn.Linear(256, 128)
		# self.layer2 = nn.Linear(128, 64)
		# self.layer3 = nn.Linear(64, 32)
		self.output = nn.Sequential(nn.Linear(32, 1))  # 二分类输出
		init.xavier_uniform_(self.output[0].weight)
		init.zeros_(self.output[0].bias)
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		self.self_gate = SelfGate(self.args.output_dim)
		# self.reduce_dim2 = nn.Linear(self.args.output_dim, 32)
		# self.fusion_layer = GatedFeatureFusion(2048)
		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim*4,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
	

		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double','reaction']
		# list_smiles = ['reaction','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
				Reactant_x = self.embedding(Reactant)
				Reactant_a = Reactant_x.permute(0,2,1).float()
				Reactant_x = self.conv_embedding(Reactant_a)
				Reactant_xc = Reactant_x.permute(0,2,1).contiguous()
				# Reactant_xc = self.self_gate(Reactant_xc)
				Reactant_x = self.pe(Reactant_xc)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0
		Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim).squeeze(-1)
		Reactant_z = self.batch_norm(Reactant_z)
		Reactant_z = self.resblock1(Reactant_z)
		Reactant_z = self.resblock2(Reactant_z)
		Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)
		return bind
class ChemAHNet_ddG_RCIM(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim1 = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.reduce_dim2 = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.reduce_dim3 = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)

		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)
        # 定义 Softmax 激活函数

		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.output = nn.Sequential(nn.Linear(32, 1))  # 二分类输出
		init.xavier_uniform_(self.output[0].weight)
		init.zeros_(self.output[0].bias)
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim*4,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
		self.self_gate = SelfGate(self.args.output_dim)
	

		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double','reaction']
		# list_smiles = ['reaction','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
		# Solvent_SMILES = (Solvent_SMILES != 1).long()
		# Reactant_SMILES = (Reactant_SMILES != 1).long()
		# B, L  = Reactant.shape
		# print(smiles,B,L)

				Reactant_a = self.embedding(Reactant)
				Reactant_x = self.reduce_dim1(Reactant_a)
				Reactant_x = self.pe(Reactant_x)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0
				Reactant_x5 = self.reduce_dim2(Reactant_a)
				Reactant_x5 = self.pe(Reactant_x5)
				Reactant_z5 = self.tx_encoder(
					x=Reactant_x5,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z5[Reactant_mask == 0] = 0


				Reactant_x7 = self.reduce_dim3(Reactant_a)
				Reactant_x7 = self.pe(Reactant_x7)
				Reactant_z7 = self.tx_encoder(
					x=Reactant_x7,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z7[Reactant_mask == 0] = 0


		Reactant_z = torch.cat([Reactant_x,Reactant_x5,Reactant_x7], dim=1)
		# Reactant_z = Reactant_z + Reactant_z5 + Reactant_z7
		# Reactant_z = Reactant_z.permute(0, 2, 1)
		# Reactant_z = torch.sum(Reactant_z, dim=2)
		# Reactant_z = self.reduce_dim2(Reactant_z)
		Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim*3).squeeze(-1)
		Reactant_z = self.batch_norm(Reactant_z)
		Reactant_z = self.resblock1(Reactant_z)
		Reactant_z = self.resblock2(Reactant_z)
		Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)

		# bind = torch.sigmoid(bind)
		# for smiles in list_smiles:
		# if smiles =='double':
		# 	x_smiles = batch[smiles].squeeze(dim=1).to(self.args.device, dtype=torch.float32)
		# 	# print(x.size())
		# 	# x = self.fusion_layer(Reactant_z, x)
		# 	# x = torch.cat([Reactant_z, x], axis=1)
		# 	x = self.resblock4(x_smiles)
		# 	x = self.resblock5(x)
		# 	x = self.resblock6(x)
		# 	# x = torch.relu(self.layer1(x))
		# 	# x = torch.relu(self.layer2(x))
		# 	# x = torch.relu(self.layer3(x))
		# 	x = self.output(x)
		# 	x = self.sigmoid(x)  # 输出为概率
		# # x = 0.6*bind + 0.4*x
		return bind
class ChemAHNet_ddG_MIM(nn.Module):
	def __init__(self, args,vocab_size,embedding_dim):
		super().__init__()
		self.args =args
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.output_type = ['infer', 'loss']
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.args.device)
		self.reduce_dim = nn.Linear(self.embedding_dim, self.args.output_dim)
		self.pe = PositionalEncoding(self.args.output_dim, max_len=self.args.max_length).to(self.args.device)
		self.bind = nn.Sequential(
			nn.Linear(32, 1),
		).to(self.args.device)
		init.xavier_uniform_(self.bind[0].weight)
		init.zeros_(self.bind[0].bias)
        # 定义 Softmax 激活函数

		self.conv_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=3,stride=1,padding=1, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.conv5_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=5,stride=1,padding=2, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv5_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)

		self.conv7_embedding = nn.Sequential(
			Conv1dBnRelu(self.embedding_dim, self.args.output_dim, kernel_size=7,stride=1,padding=3, is_bn=False),
			nn.LayerNorm(self.args.max_length)
			# nn.BatchNorm1d(self.args.output_dim)
		).to(self.args.device)
		for conv_layer in self.conv7_embedding:
			if isinstance(conv_layer, nn.Conv1d):
				init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
				init.zeros_(conv_layer.bias)
		self.output = nn.Sequential(nn.Linear(32, 1))  # 二分类输出
		init.xavier_uniform_(self.output[0].weight)
		init.zeros_(self.output[0].bias)
		self.batch_norm = nn.BatchNorm1d(self.args.output_dim)
		self.resblock1 = ResidualBlock(256, 128,0.1)
		self.resblock2 = ResidualBlock(128, 64,0.1)
		self.resblock3 = ResidualBlock(64, 32,0.1)
		self.self_gate = SelfGate(self.args.output_dim)
		self.reduce_dim2 = nn.Linear(self.args.output_dim, 32)
		# self.fusion_layer = GatedFeatureFusion(2048)
		self.tx_encoder = StandardAttentionTransformerEncoder(
			dim_model=self.args.output_dim,
			num_heads=self.args.num_heads,
			dim_feedforward=self.args.output_dim*4,
			dropout=self.args.dropout,
			norm_first=False,
			activation=F.gelu,
			num_layers=self.args.num_layers,
		).to(self.args.device)
	

		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
	def tokenfun(self, text):
		max_length = 256
		encoded_sent = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt",add_special_tokens=False)['input_ids']
		return encoded_sent
        


	def forward(self, batch):
		
		list_smiles = ['Reactant SMILES','Catalyst SMILES','Solvent SMILES','Reactants','double','reaction']
		# list_smiles = ['reaction','double']
		list_tensor = []
		
		for smiles in list_smiles:
			if smiles =='Reactants':
				Reactant = batch[smiles].long()
				Reactant_mask = (Reactant != 1).long()
				Reactant_x = self.embedding(Reactant)

				Reactant_a = Reactant_x.permute(0,2,1).float()
				Reactant_x = self.conv_embedding(Reactant_a)
				Reactant_xc = Reactant_x.permute(0,2,1).contiguous()
				# Reactant_xc = self.self_gate(Reactant_xc)
				Reactant_x = self.pe(Reactant_xc)
				Reactant_z = self.tx_encoder(
					x=Reactant_x,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z[Reactant_mask == 0] = 0
				Reactant_x5 = self.conv5_embedding(Reactant_a)
				Reactant_x5c = Reactant_x5.permute(0,2,1).contiguous()
				# Reactant_x5c = self.self_gate(Reactant_x5c)
				Reactant_x5 = self.pe(Reactant_x5c)
				Reactant_z5 = self.tx_encoder(
					x=Reactant_x5,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z5[Reactant_mask == 0] = 0


				Reactant_x7 = self.conv7_embedding(Reactant_a)
				Reactant_x7c = Reactant_x7.permute(0,2,1).contiguous()
				# Reactant_x7c = self.self_gate(Reactant_x7c)
				Reactant_x7 = self.pe(Reactant_x7c)
				Reactant_z7 = self.tx_encoder(
					x=Reactant_x7,
					# src_key_padding_mask=smiles_token_mask==0
					src_key_padding_mask=None,
				)
			# 手动恢复填充位置的输出
				Reactant_z7[Reactant_mask == 0] = 0


		# Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		# Reactant_z = torch.cat([Reactant_z,Reactant_z5,Reactant_z7], dim=1)
		Reactant_z = Reactant_z + Reactant_z5 + Reactant_z7
		Reactant_z = Reactant_z.permute(0, 2, 1)
		Reactant_z = torch.sum(Reactant_z, dim=2)
		Reactant_z = self.reduce_dim2(Reactant_z)
		# Reactant_z = F.max_pool1d(Reactant_z.permute(0, 2, 1),kernel_size=self.args.output_dim*3).squeeze(-1)
		# Reactant_z = self.batch_norm(Reactant_z)
		# Reactant_z = self.resblock1(Reactant_z)
		# Reactant_z = self.resblock2(Reactant_z)
		# Reactant_z = self.resblock3(Reactant_z)
		bind = self.bind(Reactant_z)

		return bind
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(out_features, out_features)
        
        # 定义一个 Dropout 层
        self.dropout = nn.Dropout(dropout_prob)  # Dropout 只定义一次
        
        # 如果输入和输出维度不同，使用1x1卷积或线性变换来匹配维度
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()  # 如果维度相同，就不需要做任何修改
	


    def forward(self, x):
        identity = self.shortcut(x)  # 获取输入的残差部分
        
        # 经过第一层，激活，并应用 Dropout
        out = self.layer1(x)
        out = F.relu(out)
        out = self.dropout(out)  # 使用相同的 Dropout 层

        # 经过第二层，激活，并应用 Dropout
        out = self.layer2(out)
        out = self.dropout(out)  # 再次使用相同的 Dropout 层
        
        # 添加残差连接
        out += identity
        
        return F.relu(out)  # 对最终的输出进行激活
class GatedFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFeatureFusion, self).__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, feature1, feature2):
        # Concatenate the features along the last dimension
        concatenated_features = torch.cat((feature1, feature2), dim=-1)
        # Generate the gate values
        gate = self.gate_layer(concatenated_features)
        # Weighted sum of the two features based on the gate
        fused_feature = gate * feature1 + (1 - gate) * feature2
        return fused_feature
	
class SelfGate(nn.Module):
    def __init__(self, input_dim):
        super(SelfGate, self).__init__()
        # 定义一个可学习的参数矩阵
        self.gate_weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # 初始化参数
        nn.init.xavier_uniform_(self.gate_weight)

    def forward(self, x):
        # x 是输入张量，形状为 (batch_size, sequence_length, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # 将 x 重塑为 (batch_size * sequence_length, input_dim) 以便进行矩阵乘法
        x_reshaped = x.view(-1, input_dim)

        # 计算门控值
        gate = torch.sigmoid(torch.matmul(x_reshaped, self.gate_weight))

        # 将门控值重塑回原始形状 (batch_size, sequence_length, input_dim)
        gate = gate.view(batch_size, seq_len, input_dim)

        # 应用门控
        output = x * gate
        return output



	

