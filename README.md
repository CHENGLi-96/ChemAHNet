Code for our paper "Chemistry-Informed Deep Learning Models for Predicting Stereoselectivity and Absolute Structures in Asymmetric Hydrogenation" 

This repository contains a legacy version of our implementation, which we used to produce the results in our paper.
Although the code style may be improved, we leave it unchanged for reproducibility.

# Dependencies
The code has been tested using the following environment:
* python 3.9.19
* numpy 1.24.3
* pytorch 2.0.0
* rdkit 2024.03.5

# Dataset
Experimentally reported dataset of olefin asymmetric hydrogenation reactions, compiled by Hong Xing and his collaborators, encompasses standardized information on 12,619 literature transformations conducted between 2000 and 2020. You can find information about reaction of Asymmetric Hydrogenation over there(http://asymcatml.net/). 

# Examples
## Data cleaning and labeling
The detailed workflow of data cleaning and labeling is provided in Supplementary Section 1, with executable code available in `data_cleaning_labeling.ipynb`.

## Data Preprocessing
In this study, the input to the deep learning model consists of the SMILES representations of the reactant, solvent, and catalyst.
These reaction SMILES can be tokenized through `preprocess.py`, but we leave them unchanged for reproducibility.

If the training set is in data/train.csv, call 
`process("data/train")`
defined in `preprocess.py`. 


## Description of Python Scripts and Their Functions

`main_Majorgit.py` – This script trains a model for predicting the absolute configuration of the major enantiomer in asymmetric hydrogenation of olefins.

`main_ddGgit.py` – This script trains a regression model for predicting ddG values, enabling simultaneous prediction of the stereoselectivity and absolute configuration in asymmetric hydrogenation of olefins.

`main_10fold.py` –  This script performs 10-fold cross-validation on the dataset for Rh/BINOL-phosphite-catalyzed hydrogenation of trisubstituted olefins, assessing the predictive performance of ChemAHNet.

`main_ddG_thiol.py` – This script trains and validates multiple enantioselectivity prediction tasks on Denmark’s dataset of chiral phosphoric acid (CPA)-catalyzed thiol addition to N-acylimine, assessing the predictive performance of ChemAHNet.


## Training

You can train the model using the following command:

```bash
python main_Majorgit.py --local_rank 0 --train \
--batch_size 128 --dropout=0.2 --num_heads 8 --num_layers 6 \
--embed_dim 256 --max_length 256 --output_dim 256 \
--prefix data --name tmp --epochs 250
```
Alternatively, you can also use the provided shell script:
bash train_Major_train.sh      # for the Major classification model
bash train_ddG_train.sh        # for the ddG regression model


## Testing
To evaluate a trained model, run:
```bash
python main_Majorgit.py --local_rank 0  --test \
--batch_size 128 --dropout=0.2 --num_heads 8 --num_layers 6 --embed_dim 256 --max_length 256 --output_dim 256 \
--prefix data --name tmp --epochs 250
```
Or simply:
bash train_Major_test.sh       # for the Major classification model
bash train_ddG_test.sh         # for the ddG regression model

These scripts automatically activate the conda environment, run the proper configuration, and write logs to files.
## ChemAHNet Explainability

`ChemAHNet_SHAP_Analysis.ipynb` # This notebook analyzes ChemAHNet's predictions using SHAP (SHapley Additive exPlanations) to interpret feature contributions.
