import argparse
import multiprocessing
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from piano import Composer, time_code

try:
    import rapids_singlecell as rsc
    use_rapids = True
    print('Using rapids singlecell to speed up postprocessing', flush=True)
except:
    print('Warning: Unable to use rapids singlecell in this environment', flush=True)
    use_rapids = False

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

parser = argparse.ArgumentParser(description="Run PIANO pipeline")
parser.add_argument('--rach2', action='store_true', help="Piano Concerto No. 2 in C minor, Op. 18")

# Run I/O parameters
parser.add_argument("--version", type=str, default='0.0', help="Name of run")
parser.add_argument("--adata_path", type=str, help="Path to AnnData file")
parser.add_argument("--outdir", type=str, help="Path to output directory")

# Model parameters
parser.add_argument("--n_top_genes", type=int, default=4096, help="Number of highly variable genes")
parser.add_argument("--categorical_covariate_keys", type=str, nargs='*', default=[], help="Categorical covariates to regress out")
parser.add_argument("--continuous_covariate_keys", type=str, nargs='*', default=[], help="Continuous covariates to regress out")

# Training parameters
parser.add_argument("--max_epochs", type=int, default=200, help="Max number of training epochs")

# Validation parameters
parser.add_argument("--batch_key", type=str, help="Batch key for HVG selection")
parser.add_argument("--umap_labels", nargs='*', type=str, help="Colors for UMAPs")

args = parser.parse_args()
if args.rach2:
    args.rach2 = 'Piano Concerto No. 2 in C minor, Op. 18'
    print("A Monsieur Sergei Rachmaninoff")
    print(vars(args))


# Run parameters
run_name = f'piano_v{args.version}'
outdir = f'{args.outdir}/piano/{run_name}'
os.makedirs(f'{outdir}/integration_results', exist_ok=True)
os.makedirs(f'{outdir}/figures', exist_ok=True)

# Adjustable parameters
num_workers = 0  # Set to 0 if using 'GPU', otherwise ~11 workers
memory_mode = 'GPU'  # Set to 'CPU' if no GPU available
n_neighbors = 15  # Used for rsc.pp.neighbors for UMAP
random_state = 0

# Metadata
batch_key = args.batch_key
umap_labels = args.umap_labels

# Run pipeline
with time_code('Load Anndata'):
    adata = sc.read_h5ad(args.adata_path)
    print(f"Training on: {adata}")

with time_code('Initialize Pianist'):
    pianist = Composer(
        adata,
        categorical_covariate_keys = args.categorical_covariate_keys,
        continuous_covariate_keys = args.continuous_covariate_keys,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=batch_key,
        max_epochs=args.max_epochs,
        run_name=run_name,
        outdir=outdir,
        memory_mode=memory_mode,
    )

# Run pipeline
print(f'Training and validating on {args.adata_path}')
num_cores = multiprocessing.cpu_count()
print(f'Number of CPU cores: {num_cores}', flush=True)
num_gpus = torch.cuda.device_count()
print(f'Number of GPUs: {num_gpus}', flush=True)
cuda_available = torch.cuda.is_available()
print(f'CUDA GPUs available: {cuda_available}', flush=True)

with time_code('Run pipeline'):
    pianist.run_pipeline()

with time_code('Save pianist'):
    pianist.save(f'{outdir}/pianist.pkl')

with time_code('Get Latent Representation'):
    adata.obsm['X_PIANO'] = pianist.get_latent_representation()

with time_code('Create Latent Representation DataFrame'):
    X_PIANO_df = pd.DataFrame(adata.obsm['X_PIANO'], index=adata.obs_names)

with time_code('Save Latent Space CSV'):
    X_PIANO_df.to_csv(f'{outdir}/integration_results/X_PIANO.csv')

with time_code('Computing Neighbors'):
    if use_rapids:  # GPU acceleration
        rsc.pp.neighbors(
            adata, n_neighbors=n_neighbors,
            n_pcs=pianist.model.latent_size,
            use_rep='X_PIANO',
            random_state=random_state,
        )
    else:
        sc.pp.neighbors(
            adata, n_neighbors=n_neighbors,
            n_pcs=pianist.model.latent_size,
            use_rep='X_PIANO',
            random_state=random_state,
        )

with time_code('Computing UMAP'):
    if use_rapids:  # GPU acceleration
        rsc.tl.umap(adata, random_state=random_state)
    else:
        sc.tl.umap(adata, random_state=random_state)


with time_code('Creating UMAP DataFrame'):
    UMAP_df = pd.DataFrame(
        adata.obsm['X_umap'],
        index=adata.obs_names,
    )

with time_code('Saving UMAP CSV'):
    UMAP_df.to_csv(f'{outdir}/integration_results/UMAP_2D_df.csv')

with time_code('Possibly saving Anndata'):
    adata.write_h5ad(f'{outdir}/integration_results/adata_integrated.h5ad')

# Plot UMAPs
with time_code('Plotting UMAPs'):
    print(adata, flush=True)
    for umap_label in umap_labels:
        fig = sc.pl.umap(
            adata[np.random.permutation(np.arange(adata.shape[0]))],
            color=umap_label,
            return_fig=True,
        )
        legend = fig.axes[0].get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((0.5, -0.1))  # Adjust legend coordinates
            legend.set_loc('upper center')          # Actually positions legend below
        fig.savefig(
            f'{outdir}/figures/UMAP_{umap_label}.png', bbox_inches='tight'
        )
        plt.show()
        plt.close(fig)
