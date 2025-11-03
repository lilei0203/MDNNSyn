# MDNNSynergy
In the "Comparison between different types of models" experiment of the study "A Review of Deep Learning Approaches for Drug Synergy Prediction in Cancer", MDNNSyn is retrained using the DrugComb dataset, and the hyperparameters are readjusted to ensure a fair comparison under consistent experimental conditions:

*batchsize=32
*learning rate=0.0001
*epoch=2000 

# Requirements
* python 3.7
* deepchem >= 2.5
* numpy >= 1.19
* pandas >= 1.3
* pytorch >= 1.8.0
* pytorch geometric >= 2.0.0 
* scikit-learn >= 1.0.2
* rdkit >= 2020.09.1

# Usage
```sh
  cd Model/MDNNSyn
  # for classification experiment
  python main.py
  # for regression experiment
  python main_reg.py
```
