import argparse
import gc
import multiprocessing
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
from piano import Composer, time_code, highly_variable_genes, faiss_umap, plot_umaps


try:
    import rapids_singlecell as rsc
    # sc.pp.neighbors = rsc.pp.neighbors
    # sc.tl.umap = rsc.tl.umap
    # print('Using rapids singlecell to speed up pca, neighbors, and umap', flush=True)
except:
    print('Warning: Unable to use rapids singlecell in this environment', flush=True)
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_float32_matmul_precision('high')
sc.settings.n_jobs = -1


def main(args):
    # Run parameters
    run_name = f'piano_v{args.version}'
    outdir = f'{args.outdir}/piano/{run_name}'
    os.makedirs(f'{outdir}/integration_results', exist_ok=True)
    os.makedirs(f'{outdir}/figures', exist_ok=True)

    # Adjustable parameters
    memory_mode = args.memory_mode  #'GPU'  # Set to 'CPU' if no GPU available
    num_workers = 0 if 'GPU' in memory_mode else 11  # Set to 0 if using 'GPU' or 'SparseGPU', otherwise ~11 workers for 'CPU'
    n_neighbors = 15  # Used for (r)sc.pp.neighbors for UMAP
    random_state = args.random_seed
    n_pcs_pca = args.n_pcs_pca

    # Metadata
    batch_key = args.batch_key
    umap_labels = args.umap_labels

    print(f'Number of CPU cores: {multiprocessing.cpu_count()}, Number of GPUs: {torch.cuda.device_count()}, CUDA GPUs available: {torch.cuda.is_available()}', flush=True)

    with time_code('Loading data'):
        if isinstance(args.adata_train_list, list):
            adata_train_list = [sc.read_h5ad(_) for _ in args.adata_train_list]
        else:
            adata_train_list = [sc.read_h5ad(args.adata_train_list)]

        print(f"Training on: {adata_train_list}")
        with time_code('HVG selection (Seurat v3)'):
            if args.geneset_path is None:
                if args.n_top_genes > 0:
                    print("  - Finding highly variables genes across all training data!")
                    highly_variable_genes(adata_train_list, n_top_genes=args.n_top_genes, batch_key=batch_key, subset=False)
                else:
                    for adata_train in adata_train_list:
                        adata_train.var['highly_variable'] = True
                    print("  - Using all genes!")
            else:
                var_names = np.intersect1d(adata_train_list[0].var_names, pd.read_csv(args.geneset_path, header=None).values.ravel())
                print(f"  - The {len(var_names)} genes used (and set as highly_variable) are: {var_names}")
                for _ in adata_train_list:
                    _.var['highly_variable'] = np.isin(_.var_names, var_names)
                del var_names
        with time_code("Loading validation data"):
            print("Warning: Validation data using metadata from training data for highly variable genes")
            same_valid_data_as_first_train_data, same_valid_data_as_full_train_data = False, False
            if args.adata_valid == args.adata_train_list[0]:
                same_valid_data_as_first_train_data = True
                if len(args.adata_train_list) == 1:
                    print("  - Using same validation data as training data")
                    same_valid_data_as_full_train_data = True
                else:
                    print("  - Using same validation data as first training data")
            if same_valid_data_as_first_train_data:
                adata_valid = adata_train_list[0]  # Reference to the first training adata
            else:
                adata_valid = sc.read_h5ad(args.adata_valid)  # Different dataset
                adata_valid.var['highly_variable'] = adata_train_list[0].var['highly_variable']  # Delay subsetting to HVGs until after initial PCA plots, which use the full transcriptome

    if args.plot_unintegrated:
        with time_code('Original data: PCA & UMAP'):
            adata_norm = adata_valid.copy()  # Avoid modifying original data
            sc.pp.normalize_total(adata_norm, target_sum=1e4)
            sc.pp.log1p(adata_norm)
            adata_norm = adata_norm[:, adata_norm.var['highly_variable']].copy()  # Subset to save memory
            sc.pp.pca(adata_norm, n_comps=n_pcs_pca, use_highly_variable=False)  # Avoid using hvg mask
            adata_valid.obsm['X__Original__PCA'] = adata_norm.obsm['X_pca']; del adata_norm
            sc.pp.neighbors(adata_valid, n_neighbors=n_neighbors, n_pcs=n_pcs_pca, use_rep='X__Original__PCA', random_state=random_state)
            sc.tl.umap(adata_valid, random_state=random_state)
            adata_valid.obsm['X__Original__PCA__UMAP'] = adata_valid.obsm['X_umap']; del adata_valid.obsm['X_umap'], adata_valid.uns['umap'], adata_valid.obsp['distances'], adata_valid.obsp['connectivities'], adata_valid.uns['neighbors']
            plot_umaps(adata_valid, umap_labels=umap_labels, outdir=f'{outdir}/figures', umap_key='X__Original__PCA__UMAP')

    with time_code('Subset data to training genes'):
        adata_train_list = [_[:, _.var['highly_variable']].copy() for _ in adata_train_list]
        if same_valid_data_as_first_train_data:
            adata_valid = adata_train_list[0]  # Reference to the first training adata
        else:
            adata_valid = adata_valid[:, adata_valid.var['highly_variable']].copy()

    with time_code('Training PIANO model'):
        pianist = Composer(
            adata_train_list,
            # Composer arguments
            memory_mode=memory_mode,  # Can select from ['GPU', 'SparseGPU', 'CPU', 'SparseCPU'], trading off speed for memory utilization
            compile_model=True,  # Requires GPU compatible with torch.compile
            categorical_covariate_keys=args.categorical_covariate_keys,
            continuous_covariate_keys=args.continuous_covariate_keys,
            # Gene selection
            n_top_genes=-1,  # Set to -1, as we have already subset to the top highly variable genes above, so we use all remaining genes
            hvg_batch_key=batch_key,
            # Model kwargs
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            latent_size=args.latent_size,
            adversarial=(args.adversarial == 'True'),
            distribution=args.distribution,
            parameterization=args.parameterization,
            # Training
            stratify_column=args.stratify_column,
            samples_per_class=args.samples_per_class,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            max_kld_weight=args.max_kld_weight,
            min_adv_weight=args.min_adv_weight,
            max_adv_weight=args.max_adv_weight,
            n_annealing_epochs=args.n_annealing_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=num_workers,
            early_stopping=(args.early_stopping == 'True'),
            min_delta=args.min_delta,
            patience=args.patience,
            deterministic=(args.deterministic == 'True'),  # If using compiled mode, not fully deterministic even if this parameter is set
            random_seed=args.random_seed,  # If using compiled mode, not fully deterministic even if this parameter is set
            run_name=run_name,
            outdir=outdir,
        )
        pianist.run_pipeline()
    pianist.save(f'{outdir}/pianist.pkl')

    with time_code('Validating PIANO model'):
        with time_code('Sample latent representation'):
            adata_valid.obsm['X__Original__PIANO'] = pianist.get_latent_representation(None if same_valid_data_as_full_train_data else adata_valid)
        with time_code('Run FAISS UMAP'):
            adata_valid.obsm['X__Original__PIANO__UMAP'] = faiss_umap(adata_valid.obsm['X__Original__PIANO'])
        # Check UMAP distribution
        x = adata_valid.obsm['X__Original__PIANO__UMAP'][:, 0]
        y = adata_valid.obsm['X__Original__PIANO__UMAP'][:, 1]
        x_pct = np.percentile(x, [0, 0.1, 0.5, 1, 5, 50, 95, 99, 99.5, 99.9, 100])
        y_pct = np.percentile(y, [0, 0.1, 0.5, 1, 5, 50, 95, 99, 99.5, 99.9, 100])
        print(x_pct)
        print(y_pct)
        plot_umaps(adata_valid, umap_labels=umap_labels, outdir=f'{outdir}/figures', umap_key='X__Original__PIANO__UMAP', x_outlier_pct_threshold=0.5, y_outlier_pct_threshold=0.5)

    if args.plot_counterfactual:
        with time_code('Counterfactual analysis'):
            with time_code('Compute Counterfactual PIANO UMAPs'):
                adata_cf = ad.AnnData(
                    X=pianist.get_counterfactual(None if same_valid_data_as_full_train_data else adata_valid),
                    obs=adata_valid.obs[np.unique(args.categorical_covariate_keys + args.continuous_covariate_keys + umap_labels)].copy(),  # Copy only unique, relevant columns for dataloader and umap plotting; the .copy() is probably not necessary
                    var=pd.DataFrame(index=adata_valid.var_names.copy()),  # Do not modify reference to .var; the .copy() is probably not necessary
                )
                adata_valid.obsm['X__Counterfactual__PIANO'] = pianist.get_latent_representation(adata_cf)
                sc.pp.neighbors(adata_valid, n_neighbors=n_neighbors, n_pcs=pianist.model.latent_size, use_rep='X__Counterfactual__PIANO', random_state=random_state)
                sc.tl.umap(adata_valid, random_state=random_state)
                adata_valid.obsm['X__Counterfactual__PIANO__UMAP'] = adata_valid.obsm['X_umap']; del adata_valid.obsm['X_umap'], adata_valid.uns['umap'], adata_valid.obsp['distances'], adata_valid.obsp['connectivities'], adata_valid.uns['neighbors']
                plot_umaps(adata_valid, umap_labels=umap_labels, outdir=f'{outdir}/figures', umap_key='X__Counterfactual__PIANO__UMAP')
            with time_code('Compute Counterfactual PCA UMAPs'):
                sc.pp.normalize_total(adata_cf, target_sum=1e4)
                sc.pp.log1p(adata_cf)
                sc.pp.pca(adata_cf, n_comps=n_pcs_pca, use_highly_variable=False)  # Avoid using hvg mask
                adata_valid.obsm['X__Counterfactual__PCA'] = adata_cf.obsm['X_pca']; del adata_cf
                sc.pp.neighbors(adata_valid, n_neighbors=n_neighbors, n_pcs=n_pcs_pca, use_rep='X__Counterfactual__PCA', random_state=random_state)
                sc.tl.umap(adata_valid, random_state=random_state)
                adata_valid.obsm['X__Counterfactual__PCA__UMAP'] = adata_valid.obsm['X_umap']; del adata_valid.obsm['X_umap'], adata_valid.uns['umap'], adata_valid.obsp['distances'], adata_valid.obsp['connectivities'], adata_valid.uns['neighbors']
                plot_umaps(adata_valid, umap_labels=umap_labels, outdir=f'{outdir}/figures', umap_key='X__Counterfactual__PCA__UMAP')

    # Save integration results
    print(f"Final integrated data: {adata_valid}")
    if args.save_adata:
        with time_code('Saving Anndata'):
            adata_valid.write_h5ad(f'{outdir}/integration_results/adata_integrated.h5ad')

    if args.scib_benchmarking:
        with time_code('Integration Benchmarking'):
            bm = Benchmarker(
                adata_valid, batch_key=batch_key, label_key=args.celltype,
                embedding_obsm_keys=[_ for _ in ['X__Original__PCA', 'X__Original__PIANO', 'X__Counterfactual__PCA', 'X__Counterfactual__PIANO'] if _ in adata_valid.obsm],
                pre_integrated_embedding_obsm_key='X__Original__PCA',
                bio_conservation_metrics=BioConservation(isolated_labels=False, nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans=False, silhouette_label=False, clisi_knn=False),
                batch_correction_metrics=BatchCorrection(silhouette_batch=False, ilisi_knn=True, kbet_per_label=True, graph_connectivity=False, pcr_comparison=False),
                n_jobs=-1,
            )
            bm.benchmark()
            unscaled_bm_df = bm.get_results(min_max_scale=False).T
            unscaled_bm_df.to_csv(f'{outdir}/integration_results/bm_df.csv')
            print(unscaled_bm_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PIANO pipeline")
    parser.add_argument('--rach2', action='store_true', help="Piano Concerto No. 2 in C minor, Op. 18")

    # Run I/O parameters
    parser.add_argument("--version", type=str, default='0.0', help="Name of run")
    parser.add_argument("--adata_train_list", type=str, nargs='+', help="Path(s) to AnnData file(s)")
    parser.add_argument("--adata_valid", type=str, help="Path to AnnData file")
    parser.add_argument("--outdir", type=str, help="Path to output directory")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--memory_mode", type=str, default='GPU', help="Memory mode. Default = 'GPU'")

    # Model parameters
    parser.add_argument("--n_top_genes", type=int, default=4096, help="Number of highly variable genes")
    parser.add_argument("--n_hidden", type=int, default=256, help="Number of nodes per hidden layer. Default = 256")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of hidden layers. Default = 3")
    parser.add_argument("--latent_size", type=int, default=32, help="Number of latent dimensions. Default = 32")
    parser.add_argument("--categorical_covariate_keys", type=str, nargs='*', default=[], help="Categorical covariates to regress out")
    parser.add_argument("--continuous_covariate_keys", type=str, nargs='*', default=[], help="Continuous covariates to regress out")
    parser.add_argument("--distribution", type=str, default='nb', help="Distribution for log-likelihood. Default = 'nb'")
    parser.add_argument("--parameterization", type=str, default='ksi-psi', help="Parameterization for negative binomial. Default = 'ksi-psi'")

    # Training parameters
    parser.add_argument("--stratify_column", type=str, default=None, help="Column for stratified training")
    parser.add_argument("--samples_per_class", type=int, default=1000, help="Max samples per stratification class per epoch")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of cells per mini-batch update")
    parser.add_argument("--max_kld_weight", type=float, default=0.25, help="Max KLD beta-annealing weight. Default = 0.25")
    parser.add_argument("--min_adv_weight", type=float, default=1.00, help="Min ADV beta-annealing weight. Default = 1.00")
    parser.add_argument("--max_adv_weight", type=float, default=1.00, help="Max ADV beta-annealing weight. Default = 1.00")
    parser.add_argument("--n_annealing_epochs", type=int, default=200, help="Number of epochs for beta annealing. Default = 200")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay. Default = 0")
    parser.add_argument("--early_stopping", type=str, default='False', help="Use early stopping (True/False). Default = False.")
    parser.add_argument("--min_delta", type=float, default=1.0, help="Minimum improvement over previous early stopping improvement. Default = 1.0")
    parser.add_argument("--patience", type=int, default=5, help="Max number of epochs before improvement over min_delta.")

    parser.add_argument("--adversarial", type=str, default='True', help="Use adversarial training (True/False). Default = True.")
    parser.add_argument("--deterministic", type=str, default='False', help="Use deterministic training (True/False). Default = False.")

    # Validation parameters
    parser.add_argument("--batch_key", type=str, help="Batch key for HVG selection")
    parser.add_argument("--geneset_path", type=str, default=None, help="Path to gene set to use instead of HVGs. Takes priority over HVGs.")
    parser.add_argument("--umap_labels", nargs='*', type=str, help="Colors for UMAPs")

    # Pipeline parameters
    parser.add_argument('--plot_unintegrated', action='store_true', help="Plot UMAPs of PCA of unintegrated gene expression")
    parser.add_argument('--plot_counterfactual', action='store_true', help="Plot UMAPs of PCA of counterfactual (batch-corrected) gene expression")
    parser.add_argument('--n_pcs_pca', type=int, default=50, help="Number of PCs to use for PCA")
    parser.add_argument('--scib_benchmarking', action='store_true', help="Run integration benchmarking")
    parser.add_argument('--celltype', type=str, default='Group', help="Run integration benchmarking on cell type")

    # Script parameters
    parser.add_argument('--save_adata', action='store_true', help="Save integrated adata")
    args = parser.parse_args()

    if args.rach2:
        args.rach2 = 'Piano Concerto No. 2 in C minor, Op. 18'
        print(f"A Monsieur Sergei Rachmaninoff: {vars(args)}")

    main(args)
