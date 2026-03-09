PIANO: Probabilistic Inference Autoencoder Networks for multi-Omics  
Copyright (C) 2025 Ning Wang  

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.  

This program is distributed in the hope that it will be useful,  
but WITHOUT ANY WARRANTY; without even the implied warranty of  
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the  
GNU General Public License for more details.  

You should have received a copy of the GNU General Public License  
along with this program.  If not, see <https://www.gnu.org/licenses/>.  

# README
This repository contains the source code for PIANO: Probabilistic Inference Autoencoder Networks for multi-Omics.

## Installation:
Create an uv environment as follows (strongly recommended):
```
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'cache-dir = "/path/to/cache/directory/uv_cache"' >>~/.config/uv/uv.toml

# Create uv environment
uv venv --python 3.11.14
source .venv/bin/activate
uv pip install piano-integration[rapids]

# If not using rapids single cell, use the following:
uv pip install piano-integration

If you have issues with installation, you can add the following flag: ` --index-strategy unsafe-best-match`, e.g.
`uv pip install piano-integration[rapids] --index-strategy unsafe-best-match`
`uv pip install piano-integration --index-strategy unsafe-best-match`
`uv pip install piano-integration[all] --index-strategy unsafe-best-match`

# Alternative installation methods:
`uv pip install .[rapids]`  # Using rapids
`uv pip install .`  # Not using rapids
`uv pip install -r requirements_rapids.txt --index-strategy unsafe-best-match`  # Including rapids
`uv pip install -r requirements_lite.txt --index-strategy unsafe-best-match`  # Minimal installation
`uv pip install -r requirements.txt --index-strategy unsafe-best-match`  # Containing all optional libraries
```

Or, create a conda environment as follows (much slower than uv):
```
# Install miniconda if not already installed
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Create conda environment in a new terminal
conda create -n piano python=3.11.14 -y
conda activate piano
pip install piano-integration[rapids]

# Similar options are available as those listed above for uv
```

### Triton (torch.compile) compilation for ARM architectures:
```
# Instructions for compiling triton from scratch
git clone https://github.com/triton-lang/triton.git
cd triton
uv pip install -r python/requirements.txt # Using uv installed for ARM
uv pip install -e .
```

## Code Overview:
PIANO is both an integration method and a generative model. Along with obtaining a latent space for scRNAseq raw counts data, PIANO is able to perform "batch" correction on the gene expression data itself. These can be obtained via the get_latent_representation() and get_counterfactual() functions. Moreover, the get_counterfactual() function also supports predicting how gene expression may be influenced by categorical and/or continuous covariates. The default functionality of the  get_counterfactual() function is to perform batch-correction by setting the covariates = "marginal", but specific combinations can be specified using a dictionary mapping covariates to their intended values (e.g., changing a categorical species covariate from mouse to human or changing a z-scored continuous covariate such as ag: {'species': 'Human', 'age': 1.5 }).

The most important integration parameters are the weights that dictate the strength of the integration. For adjusting the strength of batch correction, the min_adv_weight and max_adv_weight are available for fine-tuning when adversarial mode is set to True. The KL Divergence weight can similarly be adjusted with min_kld_weight and max_kld_weight, which also increases batch correction but is more likely to over over-correct. It is recommended to use a lower KL divergence weight (i.e., reduce from 0.25 to 0.125) if the latent space experiences over-correction, such as smaller, noisier datasets, or datasets with few genes (e.g., spatial transcriptomic atlases). Alternatively, you can use stronger weights and perform a hierarchical integration, subsetting the data at each level of the integration to avoid losing biological conservation.

### Pipeline:
A simple use case is provided in run_piano_integration.py  
You can call run this script as follows:
```
python3 run_piano_integration.py \
  --version 0.0_piano_integration \
  --adata_path /path/to/adata.h5ad \
  --outdir ../results \
  --categorical_covariate_keys your covariates here \
  --batch_key your_primary_hvg_batch_key_here \
  --umap_labels your umap labels here
```
You can add additional command line arguments using argparse or modifying directly in the script to customize the parameters used for training.
The Composer class provides an easy-to-use interface for performing data integration and generative modeling.
The PIANO pipeline works as follows:
1. First, the model hyperparameters are specified. The most important parameters are the model size, covariates, training epochs, and KL divergence weight. 
  - Highly variable gene selection can be handled either before passing in the data (an AnnData or list of AnnDatas) to the Composer class. PIANO supports calculating highly variable genes using a distributed version of ScanPy's "seurat_v3" flavor. This allows feature selection over multiple, separate AnnDatas without the need to concatenate the data into a single AnnData, which provides flexibility for larger datasets (see run_pinao_integration.py for an example). Alternatively, you can also specify a path to a text file containing the genes you wish to use for analysis (one gene name per line).
2. Next, an instance of the `Composer` class is initialized to hold all the parameters for the model and training. This "pianist" calls `.run_pipeline()`, which handles the following details:
  - The pianist uses `.initialize_features()` to encode the covariates and to select highly variable genes for the data, if not already selected. Passing in -1 for "n_top_genes" will use all genes present in the data, which is helpful if the data has already been subset to highly variable genes.  
  - The pianist uses `.prepare_data()` to prepare the AnnDatasets for the data.  
  - The pianist uses `.prepare_model(**self.model_kwargs)` to prepare the `Etude` (PyTorch) model using the appropriate genes and covariates.  
  - The pianist uses `.train()` to train the `Etude` model.  
3. To retrieve the integrated latent representation on the data used for training, call `.get_latent_representation()`. You can also pass in an AnnData (that contains all of the genes used in the training data).
4. To retrieve "batch-corrected" gene counts, call `.get_counterfactual()`, which uses the default argument of covariates='marginal'. To generate a probabilistic reconstruction, call `.get_counterfactual()` with covariates=None. To specify custom combinations of categorical or continuous covariates, you can also pass in a dictionary into the covariates argument, with the keys being the covariates and the respective value for each key as the desired value. These custom covariates are one-hot encoded if categorical or z-scored if continuous (using the training data as the distribution for z-scoring), with covariates that are not specified marginalized (averaged) across categories for categorical covariates or set to 0 (z-score) for continuous covariates.
5. Finally, we visualize the results by plotting the integrated UMAPs.  

### Relevant Classes:
#### AnnDataset:
Takes in an input AnnData object and creates a PyTorch Dataset. 
The counts matrix in .X is forced into non-sparse tensor.
Then, batch columns from .obs are integer-encoded if categorical and concatenated to the counts.

#### Etude:
The Etude class is a PyTorch module that implements a variational autoencoder (VAE).
The input to the encoder are the gene counts and batch columns, which are reconstructed by the decoder.
This model differs from the original VAE by also including sets of weights for each gene using a generalized linear model (GLM).
These weights mitigate the influence of batch effects, continuous or categorical keys in the latent space representation.
For these covariates, continous values are kept in one column, while categorical keys are one-hot encoded.
These batch keys are stored as augmented columns of the AnnDataset object used for training.

#### Composer:
This class handles the pipeline of training the model, saving or loading a trained model, and retrieving integrated latent spaces.  

The Composer class parses the data to obtain how many columns to use for genes after selecting for highly variable genes using the batch_key.  
Importantly, the code supports highly variable gene selection over multiple adatas, avoiding the need to concatenate numerous large objects.  
Alternatively, you can pass in a set of genes in a text file, which is loaded using pd.read_csv(self.geneset_path, header=None).values.ravel()  

The Composer class similarly supports training the Etude model over multiple adatas without the need to concatenate them.  
Shown below are the some of the recommended parameters with descriptions. Full parameters can be found in `utils/composer`.  
Other parameters that can be toggled include the method of gene selection, covariates, and number of layers, hidden nodes, and latent dimensions.  

```
# Training data
adata: Union[ad.AnnData | List[ad.AnnData]],

# Composer arguments
memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'backed'] = 'GPU',
compile_model: bool = True,
categorical_covariate_keys=None,
continuous_covariate_keys=None,
unlabeled: str = 'Unknown',

# Gene selection
flavor: str = 'seurat_v3',  # Only Seurat V3 is supported (for multiple AnnDatas)
n_top_genes: int = 4096,
hvg_batch_key: str = None,
geneset_path: str = None,

# Model kwargs
## Architecture
n_hidden: int = 256,
n_layers: int = 3,
latent_size: int = 32,
## Training mode
adversarial: bool = True,
## Hyperparameters
dropout_rate: float = 0.1,
batchnorm_eps: float = 1e-5,       # Torch default is 1e-5
batchnorm_momentum: float = 1e-1,  # Torch default is 1e-1
epsilon: float = 1e-5,             # Torch default is 1e-5
## Distribution
distribution: Literal['nb', 'zinb'] = 'nb',

# Training
max_epochs: int = 200,
## Beta annealing
batch_size: int = 128,
min_kld_weight: float = 0.00,
max_kld_weight: float = 0.25,
min_adv_weight: float = 1.00,
max_adv_weight: float = 1.00,
n_annealing_epochs: int = None,
## Hyperparameters
lr: float = 5e-4,
weight_decay: float = 0.00,
shuffle: bool = True,
drop_last: bool = True,
num_workers: int = 0,
## Early stopping
early_stopping: bool = True,
min_delta: float = 1.00,
patience: int = 5,
## Checkpoints
save_initial_weights: bool = False,
checkpoint_every_n_epochs = None,

# Reproducibility
deterministic: bool = True,
random_seed: int = 0,

# Output
run_name: str = 'piano_integration',
outdir: str = './results/',
```

## Contact:
nw8333 at princeton dot edu
