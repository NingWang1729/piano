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
The current version of the pre-print can be found on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.06.743394v1).  

See [CHANGELOG.md](CHANGELOG.md) for what changed in each version.

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

The most important integration parameters are the weights that dictate the strength of the integration. For adjusting the strength of batch correction, the min_adv_weight and max_adv_weight are available for fine-tuning when adversarial mode is set to True. Both default to 0.5, a constant adversary weight: a stronger adversary mixes batches more but can cost label-transfer accuracy in the largest cell types, and warming it up from 0 makes little difference. On small, balanced atlases a weight of 1 can do better, so raise it if your batches stay apart. The KL Divergence weight can similarly be adjusted with min_kld_weight and max_kld_weight; it is annealed linearly from min_kld_weight to max_kld_weight (default 0.5) over n_annealing_epochs (default 200, whatever max_epochs is; None also gives the default; pass n_annealing_epochs=max_epochs to anneal over the whole run; the adversary weight, when min_adv_weight is below max_adv_weight, is warmed up over the same epochs) and then held at max_kld_weight, so training for fewer than 200 epochs ends below it (e.g. 0.25 after 100 epochs, and 0.125 after the default 50). With the default posterior-mean latents, the KL weight changes the integration little up to 0.5, while a weight of 1 mixes batches markedly less. With latents drawn from the posterior, a higher KL weight gives much more batch correction but is more likely to over-correct, so a lower weight can still help if the latent space experiences over-correction, such as smaller, noisier datasets, or datasets with few genes (e.g., spatial transcriptomic atlases); these cases were not benchmarked. Alternatively, you can use stronger weights and perform a hierarchical integration, subsetting the data at each level of the integration to avoid losing biological conservation.

Training uses mini-batches of batch_size cells (default 512). Compared with smaller batches, 512 gives about the same integration and trains several times faster per epoch; much larger batches at the default learning rate can lose batch correction. With drop_last=True (default), a dataset with fewer cells per epoch than batch_size would have no full batch, so the Composer raises a ValueError when it is created: pass a smaller batch_size, or drop_last=False to train on one smaller batch.

The optimizer is AdamW with learning rate lr=2e-4, torch's eps 1e-8, and no weight decay (weight_decay=0). Compared with torch's default 1e-3, lr 2e-4 mapped cell types across modalities slightly better on a large atlas, while 1e-3 mixed batches more on a small, balanced one: try lr=1e-3 if your batches stay apart. The choice of optimizer (Adam or AdamW) and weight decay made little difference.

Training runs max_epochs=50 epochs by default, without early stopping. On a large atlas 20-50 epochs mapped cell types best, while on a small, balanced atlas the default mixed batches least of the lengths tried: train longer (e.g. max_epochs=200) or use lr=1e-3 when batches need more mixing.

dropout_rate (default 0.1) is the dropout of the encoder layers only, and decoder_dropout_rate that of the decoder layers (default 0, no dropout; None uses dropout_rate there too); the adversary has no dropout. More decoder dropout mixes batches less; it can raise label-transfer accuracy in the largest cell types, but not the average over cell types. On small, balanced atlases decoder_dropout_rate=0.1 can do slightly better.

Training and inference (train, get_latent_representation and get_counterfactual) use TF32 matmuls by default (matmul_precision='high'; 'highest' gives full FP32). TF32 needs an Ampere or newer GPU and gives the same results as FP32 on the CPU. TF32 can train noticeably faster on GPUs when there are many genes, and changes the integration scores less than training with another seed does. The precision is set only while these methods run the model, and the caller's own setting (torch.set_float32_matmul_precision) is restored afterwards; since torch's setting is process-wide, other threads of the process see it during a call.

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
Genes are selected with `--n_top_genes` and `--geneset_path`, following the same rules as the Composer (see below): by default, 4096 highly variable genes are selected (with `--batch_key` as batches), or the whole gene set when `--geneset_path` is given; with `--geneset_path`, `--n_top_genes N` selects N highly variable genes ranked within the gene set, and `--n_top_genes -1` uses all genes (or the whole gene set).
The Composer class provides an easy-to-use interface for performing data integration and generative modeling.
The PIANO pipeline works as follows:
1. First, the model hyperparameters are specified. The most important parameters are the model size, covariates, training epochs, and KL divergence weight. 
  - Highly variable gene selection can be handled either before passing in the data (an AnnData or list of AnnDatas) to the Composer class. PIANO supports calculating highly variable genes using a distributed version of ScanPy's "seurat_v3" flavor. This allows feature selection over multiple, separate AnnDatas without the need to concatenate the data into a single AnnData, which provides flexibility for larger datasets (see run_pinao_integration.py for an example). Alternatively, you can also specify a path to a text file containing the genes you wish to use for analysis (one gene name per line).
2. Next, an instance of the `Composer` class is initialized to hold all the parameters for the model and training. This "pianist" calls `.run_pipeline()`, which handles the following details:
  - The pianist uses `.initialize_features()` to encode the covariates and to select highly variable genes for the data, if not already selected. Without a gene set (geneset_path), "n_top_genes" defaults to None, which selects 4096 highly variable genes, and passing in -1 will use all genes present in the data, which is helpful if the data has already been subset to highly variable genes. The pianist keeps its own AnnData for each training AnnData, with only X, obs and var (layers, obsm, obsp, varm and uns are dropped). In memory, an AnnData (not a view) that already has exactly the selected genes in the same order (all genes with -1 and no gene set, or the whole gene set) shares its counts (adata.X) with the pianist instead of copying them, since counts can be very large, and a warning says so when the Composer is created: do not change adata.X in place while using the pianist. Pass copy_adata=True to the Composer for a full copy with no shared memory; AnnDatas subset to HVGs (also HVGs within a gene set) or to other genes, and views, are always copied. Backed mode reads the counts from disk and is unaffected.  
  - The pianist uses `.prepare_data()` to prepare the AnnDatasets for the data.  
  - The pianist uses `.prepare_model(**self.model_kwargs)` to prepare the `Etude` (PyTorch) model using the appropriate genes and covariates.  
  - The pianist uses `.train()` to train the `Etude` model.  
3. To retrieve the integrated latent representation on the data used for training, call `.get_latent_representation()`, which returns the cells of all training AnnDatas, in the order they were passed. You can also pass in an AnnData (that contains all of the genes used in the training data, in any order; extra genes are left out), which returns exactly its own cells. An AnnData missing some of the training genes (`.var_names`) raises a ValueError, in every memory mode.
4. To retrieve "batch-corrected" gene counts, call `.get_counterfactual()`, which uses the default argument of covariates='marginal'. To generate a probabilistic reconstruction, call `.get_counterfactual()` with covariates=None. To specify custom combinations of categorical or continuous covariates, you can also pass in a dictionary into the covariates argument, with the keys being the covariates and the respective value for each key as the desired value. These custom covariates are one-hot encoded if categorical or z-scored if continuous (using the training data as the distribution for z-scoring), with covariates that are not specified marginalized (averaged) across categories for categorical covariates or set to 0 (z-score) for continuous covariates.
   `.get_latent_representation()` and `.get_counterfactual()` both take a memory_mode argument (default: the Composer's memory mode). Without a CUDA GPU, 'GPU' and 'SparseGPU' fall back to 'CPU' and 'SparseCPU' with a warning, as for the Composer. A call in 'GPU' or 'SparseGPU' uses 0 DataLoader workers, with a warning if the Composer has num_workers > 0 (only creating a GPU or SparseGPU Composer with num_workers > 0 is an error).
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
Only load Composers (`Composer.load`) from files you trust: a pickle can run code when loaded. `Composer.load` refuses files holding classes that a Composer saved by `Composer.save` does not hold (pickle.UnpicklingError), but this is not a sandbox.  

The Composer class parses the data to obtain how many columns to use for genes after selecting for highly variable genes using the batch_key.  
Importantly, the code supports highly variable gene selection over multiple adatas, avoiding the need to concatenate numerous large objects.  
Alternatively, you can pass in a set of genes in a text file (geneset_path, one gene per line with no header), which is loaded using pd.read_csv(geneset_path, header=None, dtype=str). The genes of the set found in the data are used in the AnnData's gene order. The file is read when the Composer is created: genes of the set that are not in the data are left out with a warning, and a file with no genes in the data (or no genes at all) raises a ValueError. `piano.read_gene_set_genes(adata.var_names, geneset_path)` returns the same genes, to check a gene set before training.  
With a gene set, n_top_genes selects highly variable genes from within the set: the default None (or n_top_genes <= 0) uses the whole set, and a smaller number selects that many Seurat v3 highly variable genes ranked among the gene-set genes only (the mean-variance trend is fitted on them, with hvg_batch_key as batches), in every memory mode. Ranking within the set can select different genes than ranking the whole transcriptome first and then keeping the genes in the set. A number larger than the gene-set genes found uses them all, with a warning when the Composer is created. A very small gene set may have too few variable genes to fit the trend, which raises a ValueError: use the whole set instead (n_top_genes=None). `piano.highly_variable_genes(adata, n_top_genes=..., genes=genes, inplace=False)` ranks a subset of genes the same way.  
Every training AnnData must have all selected genes (in any order). This is checked when the genes are selected (initialize_features, run by run_pipeline or prepare_data), in every memory mode, and an AnnData missing some of them raises a ValueError. Highly variable gene selection needs the same var_names in every AnnData and raises its own ValueError first.  

The Composer class similarly supports training the Etude model over multiple adatas without the need to concatenate them.  
Shown below are the some of the recommended parameters with descriptions. Full parameters can be found in `utils/composer`.  
Other parameters that can be toggled include the method of gene selection, covariates, and number of layers, hidden nodes, and latent dimensions.  

```
# Training data
adata: Union[ad.AnnData | List[ad.AnnData]],

# Composer arguments
memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] = 'GPU',  # Without a CUDA GPU, 'GPU' and 'SparseGPU' fall back to 'CPU' and 'SparseCPU' with a warning, also when a Composer saved in them is loaded
compile_model: bool = True,
categorical_covariate_keys=None,  # Each column holds one kind of value (number, str or bool), apart from missing values and the unlabeled value, else a ValueError; int mixed with float warns; a column with no labeled values (only missing values or the unlabeled value) in any training AnnData is a ValueError
continuous_covariate_keys=None,
unlabeled: str = 'Unknown',  # Cells with this value, or a missing value (NaN, None, pd.NA), in a categorical covariate have no label: their covariate values are all zeros; missing values warn once per column, at Composer creation and in get_latent_representation and get_counterfactual; a missing value given as a counterfactual categorical covariate raises (pass this value instead)

# Mode (see "scVI mode" below)
mode: Literal['piano', 'scvi'] = 'piano',
batch_key: str = None,  # scVI mode only: the batch column

# Gene selection
flavor: str = 'seurat_v3',  # Only Seurat V3 is supported (for multiple AnnDatas)
n_top_genes: int = None,  # None: 4096 HVGs, or the whole gene set with geneset_path; with a gene set, a smaller number selects that many HVGs within it; scVI mode: None keeps all genes
hvg_batch_key: str = None,
geneset_path: str = None,

# Model kwargs (settings left as None take their mode's default: PIANO's, then scVI's)
## Architecture
n_hidden: int = None,  # 512; scVI mode: 512 too (scvi-tools: 128)
n_layers: int = None,  # 4; scVI mode: 4 too (scvi-tools: 1)
latent_size: int = None,  # 32; scVI: 10
## Training mode
adversarial: bool = None,  # True; scVI: False
## Hyperparameters
dropout_rate: float = 0.1,         # Encoder layers only (the adversary has no dropout), as scvi-tools' dropout_rate
decoder_dropout_rate: float | None = 0.0,  # Decoder layers; None: same as dropout_rate
batchnorm_eps: float = 1e-5,       # Torch default is 1e-5
batchnorm_momentum: float = 1e-1,  # Torch default is 1e-1
epsilon: float = 1e-5,             # Torch default is 1e-5
## Distribution
distribution: Literal['nb', 'zinb'] = 'nb',

# Training
max_epochs: int = None,  # 50, which ends at KL weight 0.125; scVI: 400 up to 20,000 cells, then fewer in proportion
## Beta annealing
batch_size: int = None,  # 512 cells per training mini-batch, also in scVI mode (with fewer cells per epoch, lower it or set drop_last=False)
min_kld_weight: float = 0.00,
max_kld_weight: float = None,  # 0.5, annealed linearly from min_kld_weight over n_annealing_epochs, then held; scVI: 1.0
min_adv_weight: float = 0.50,  # Equal to max_adv_weight: a constant adversary weight, no warm-up
max_adv_weight: float = 0.50,
n_annealing_epochs: int = None,  # 200, fixed so training longer or shorter keeps the schedule; scVI: 400
## Hyperparameters
lr: float = None,  # 2e-4 (see above for when to try 1e-3); scVI: 1e-3
weight_decay: float = None,  # 0.0; scVI: 1e-6
shuffle: bool = True,
drop_last: bool = None,  # True; scVI: False
num_workers: int = 0,  # must be 0 for GPU and SparseGPU memory modes
## Early stopping
early_stopping: bool = None,  # False: train all max_epochs (True stops after patience epochs without min_delta gain); scVI: False
min_delta: float = None,  # 1.0; scVI: 0.0
patience: int = None,  # 5; scVI: 45
## Checkpoints
save_initial_weights: bool = False,
checkpoint_every_n_epochs = None,
## Precision
matmul_precision: Literal['highest', 'high'] | None = None,  # 'high' in both modes. Any case; TF32 ('high') or FP32 ('highest') matmuls during train and inference, restoring the caller's setting afterwards (process-wide while a call runs)

# Reproducibility
deterministic: bool | None = True,  # None: same as False
random_seed: int | None = 0,  # None: no seeding; results depend on the caller's random state

# Output
run_name: str = 'piano_integration',
outdir: str = './results/',
```

#### scVI mode
`Composer(adata, mode='scvi', batch_key='batch', categorical_covariate_keys=[...])` trains the model of scvi-tools 1.4.2's `scvi.model.SCVI` (NB likelihood by default, or ZINB with `distribution='zinb'`, as scVI's `gene_likelihood`; one dispersion per gene and batch, the batch and other categorical covariates fed to every decoder layer) with scVI's training: Adam with its large epsilon, the mean loss over cells, a KL weight that rises once per epoch over 400 epochs, 90% of the cells for training and 10% for validation, early stopping on the validation ELBO when `early_stopping=True` (off by default, as in scVI), and the last epoch's weights. Settings left as None take scVI's defaults, listed above, and users may override them. They differ from scvi-tools' in three ways, for speed and a larger network with equivalent results: 4 hidden layers of 512 units instead of scvi-tools' 1 x 128, which is rarely used as is (pass n_layers=1, n_hidden=128 for scvi-tools' own); batch_size 512 instead of 128; and TF32 matmuls instead of FP32 (matmul_precision='highest' for FP32). dropout_rate is the encoder's dropout, as in scvi-tools. It runs on PIANO's faster path (data on the GPU, compiled training step) and aims for results equivalent to scvi-tools', not identical numbers. PIANO's own code computes the model; scvi-tools is not needed (it is only used by the tests, as a reference).

Settings that cannot apply in scVI mode raise when the Composer is created: adversarial training, parameterization, continuous and sparse continuous covariates, stratified sampling, and settings without a counterpart in scVI unless left at their default: PIANO's batch-norm settings, the adversarial weights, epsilon (PIANO's minimum posterior scale; scVI has its own variance floor) and decoder_dropout_rate (scVI's decoder has no dropout layers; ZINB's "dropout" is the zero-inflation probability, a separate output head). `get_counterfactual` is not available in scVI mode yet. As in PIANO mode, cells with the unlabeled value get zeros for that covariate (scVI would treat it as a category).

## Contact:
nw8333 at princeton dot edu
