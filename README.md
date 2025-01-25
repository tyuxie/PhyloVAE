
### <div align="center"> PhyloVAE: Unsupervised Learning of Phylogenetic Trees via Variational Autoencoders <div> 
### <div align="center"> ICLR 2025 Poster <div> 

<div align="center">
  <a href="https://arxiv.org/abs/XX"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://openreview.net/forum?id=Z8TglKXDWm"><img src="https://img.shields.io/static/v1?label=Paper&message=OpenReview&color=red&logo=openreview"></a> &ensp;
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> &ensp;
</div>

<div align="center">
Tianyu Xie, Harry Richman, Jiansi Gao, Frederick A Matsen IV, Cheng Zhang
</div>

## Installation
This repository is a light CPU-based implementation of PhyloVAE.
To create the torch environment, use the following command:
```
conda env create -f environment.yml
conda activate phyloinfer
```

## Instruction for Tree Topology Density Estimation
### Training Set Construction
Before starting your training, the training dataset should be constructed by running the command
```
python -c '''
from datasets import process_data; process_data($DATASET, $REP_ID);
'''
```
and the ground truth should be constructed by running the command
```
python -c '''
from datasets import process_empFreq; process_empFreq($DATASET);
'''
```
The ```$DATASET``` is a string value refering to the name of the dataset, and the ```$REP_ID``` is an integer indicating the index of the phylogenetic analysis (since multiple analysis is a common practice to derive reliable results).

Note that these commands automatically constructs the tree topology encodings and node embeddings of DS1-8 (which are standard benchmarks in phylogenetic inference, see [VBPI](https://github.com/zcrabbit/vbpi), [ARTree](https://github.com/tyuxie/ARTree), etc).
The complete data sets of DS1-8 can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1qMdv_NxpsLZlu510izs26V6b02smGAoH?usp=sharing).
One may also manually construct their own training set, as long as there exists a tree file:
- ```data/short_run_data/$DATASET/rep_$REP_ID/$DATASET.trprobs``` or 
- ```data/raw_data/$DATASET/rep_$REP_ID/$DATASET.trprobs```.


### Training
To reproduce the result on the DS1-8 benchmarks, please run the following command.
```
python main.py base.mode=train data.dataset=$DATASET data.rep_id=$REP_ID decoder.num_layers=4 decoder.latent_dim=2 objective.batch_size=10 objective.n_particles=32 
```
You can freely choose the hyperparameters which can be crucial to the model performance.
- ```decoder.num_layers```: The number of layers in MLPs of the generative model $p_{\theta}(\tau|z)$.
- ```decoder.latent_dim```: The dimension ($d$) of the latent variable $z$.
- ```objective.batch_size```: The number of tree topology samples in a mini-batch
for stochastic optimization.
- ```objective.n_particles```: The number of particles ($K$) in the multi-sample lower bound.

You can also consider unsupervised learning on other data sets, as long as you have manually constructed your own training set (and the ground truth).

### Evaluation
Once the training is finished, you can use the following command to compute the marginal likelihood on training set by
```
python main.py base.mode=test data.dataset=$DATASET data.rep_id=$REP_ID
```
and compute the KL divergence to the ground truth by
```
python main.py base.mode=test data.dataset=$DATASET data.rep_id=$REP_ID data.empFreq=True
```

## Instruction for Building Tree Topology Representations
To obtain the representations of a set of tree topologies using PhyloVAE, consider the following steps:
1. Gather the tree topologies you are interested in in an ```.trprobs``` file, which is usually the output file of MrBayes. It should be putted at ```data/short_run_data/$DATASET/rep_1/$DATASET.trprobs```.
2. Run the command 
  ```
  python -c '''from datasets import process_data; process_data($DATASET, 1);'''
  python main.py base.mode=train data.dataset=$DATASET data.rep_id=1 decoder.num_layers=2 decoder.latent_dim=2 objective.batch_size=10 objective.n_particles=32 base.datetime=20XX-XX-XX-rep
  ```
  to construct the training set and train the PhyloVAE model on it.
3. Run the command
  ```
  python main.py base.mode=rep data.dataset=$DATASET data.rep_id=1 decoder.num_layers=2 decoder.latent_dim=2 objective.batch_size=10 objective.n_particles=32 base.datetime=20XX-XX-XX-rep
  ```
  to obtain the represetations of the tree topologies in the training set. This will outputs a ```.txt``` file which includes the 2D representations of the tree topologies in the ```.trprobs``` file. You can then choose your favorite tool to visualize the representations or use them for downstream tasks.
  
You can skip step 2 if you have a pre-trained PhyloVAE model. For more details about the training and evaluation configuratiion, please refer to **Instruction for Tree Topology Density Estimation**.


## References
When building this codebase for PhyloVAE, we refer to codes of the following two articles.
- Zhang C. *Learnable Topological Features For Phylogenetic Inference via Graph Neural Networks*. ICLR 2023.
- Zhou M Y, Yan Z, Layne E, et al. *PhyloGFN: Phylogenetic inference with generative flow networks*. ICLR 2024.

---

Please consider citing our work if you find this codebase useful:
```
@inproceedings{
xie2025phylovae,
title={Phylo{VAE}: Unsupervised Learning of Phylogenetic Trees via Variational Autoencoders},
author={Xie, Tianyu and Richman, Harry and Gao, Jiansi and Matsen IV, Frederick A and Zhang, Cheng},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
}
```