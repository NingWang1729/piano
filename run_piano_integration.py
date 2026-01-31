import argparse
import multiprocessing
import os
from memory_profiler import profile

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

from piano import Composer, time_code, highly_variable_genes

try:
    import rapids_singlecell as rsc
    sc.pp.pca = rsc.pp.pca
    sc.pp.neighbors = rsc.pp.neighbors
    sc.tl.umap = rsc.tl.umap
    print('Using rapids singlecell to speed up pca, neighbors, and umap', flush=True)
except:
    print('Warning: Unable to use rapids singlecell in this environment', flush=True)
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)


def main(args):
    # Run parameters
    run_name = f'piano_v{args.version}'
    outdir = f'{args.outdir}/piano/{run_name}'
    os.makedirs(f'{outdir}/integration_results', exist_ok=True)
    os.makedirs(f'{outdir}/figures', exist_ok=True)

    # Adjustable parameters
    memory_mode = 'GPU'  # Set to 'CPU' if no GPU available
    # memory_mode = 'CPU'  # Set to 'CPU' if no GPU available
    num_workers = 0 if memory_mode != 'CPU' else 11  # Set to 0 if using 'GPU' or 'SparseGPU', otherwise ~11 workers for 'CPU'
    n_neighbors = 15  # Used for (r)sc.pp.neighbors for UMAP
    random_state = 0
    n_pcs_pca = args.n_pcs_pca

    # Metadata
    batch_key = args.batch_key
    umap_labels = args.umap_labels

    def plot_umaps(adata, umap_labels, outdir, prefix='UMAP', show_interactive=False):
        adata_perm = ad.AnnData(obs=adata.obs[umap_labels])
        adata_perm.obsm['X_umap'] = adata.obsm['X_umap']
        adata_perm = adata_perm[np.random.permutation(np.arange(adata.shape[0]))].copy()  # Expensive, but avoids N x N sparse indexing cost

        os.makedirs(outdir, exist_ok=True)
        for umap_label in umap_labels:
            fig = sc.pl.umap(adata_perm, color=umap_label, return_fig=True)
            legend = fig.axes[0].get_legend()
            if legend is not None:
                legend.set_bbox_to_anchor((0.5, -0.1))
                legend.set_loc('upper center')
            fig.savefig(f'{outdir}/{prefix}__{umap_label}.png', bbox_inches='tight',)
            if show_interactive:
                plt.show()
            plt.close(fig)

    # Run pipeline
    print(f'Training and validating on {args.adata_train_list}')
    print(f'Number of CPU cores: {multiprocessing.cpu_count()}, Number of GPUs: {torch.cuda.device_count()}, CUDA GPUs available: {torch.cuda.is_available()}', flush=True)

    with time_code('Load data'):
        if isinstance(args.adata_train_list, list):
            adata_train_list = [sc.read_h5ad(_) for _ in args.adata_train_list]
        else:
            adata_train_list = [sc.read_h5ad(args.adata_train_list)]
        print('Adding sparsity (qc score) as a continuous covariate')
        for adata_train in adata_train_list:
            adata_train.obs['density'] = np.mean(adata_train.X > 0, axis=1)

        print(f"Training on: {adata_train_list}")
        with time_code('HVG selection (Seurat v3)'):
            if args.geneset_path is None:
                print("Finding highly variables genes across all training data!")
                highly_variable_genes(adata_train_list, n_top_genes=args.n_top_genes, batch_key=batch_key, subset=False)
            else:
                var_names = np.intersect1d(adata_train_list[0].var_names, pd.read_csv(args.geneset_path, header=None).values.ravel())
                print(f"My {len(var_names)} genes are: {var_names}")
                for _ in adata_train_list:
                    _.var['highly_variable'] = np.isin(_.var_names, var_names)
                    print(f"My {np.sum(_.var['highly_variable'])} actual used genes are: {_.var}")
                del var_names
        with time_code("Loading validation data"):
            print("Warning: Validation data using metadata from training data for highly variable genes")
            same_train_and_validation_data = False
            if len(args.adata_train_list) == 1 and args.adata_valid == args.adata_train_list[0]:
                print("Using same training and validation data")
                same_train_and_validation_data = True
            if same_train_and_validation_data:
                # Delay subsetting to HVGs until after initial PCA plots, which use full transcriptome
                adata_valid = adata_train_list[0]
            else:
                adata_valid = sc.read_h5ad(args.adata_valid)
                adata_valid.var['highly_variable'] = adata_train_list[0].var['highly_variable']
                adata_valid = adata_valid[:, adata_valid.var['highly_variable']].copy()

    if args.plot_unintegrated:
        with time_code('Original data: PCA & UMAP'):
            adata_norm = adata_valid.copy()  # Avoid modifying original data
            sc.pp.normalize_total(adata_norm, target_sum=1e4)
            sc.pp.log1p(adata_norm)
            adata_norm = adata_norm[:, adata_norm.var['highly_variable']].copy()  # Subset to save memory
            sc.pp.pca(adata_norm, n_comps=50, use_highly_variable=False)  # Avoid using hvg mask
            sc.pp.neighbors(adata_norm, n_neighbors=n_neighbors, n_pcs=n_pcs_pca, use_rep='X_pca', random_state=random_state)
            sc.tl.umap(adata_norm, random_state=random_state)
            if args.save_original_pca or args.scib_benchmarking:
                adata_valid.obsm['X__Original__PCA'] = adata_norm.obsm['X_pca']
                adata_valid.obsm['X__Original__PCA__UMAP'] = adata_norm.obsm['X_umap']
            plot_umaps(adata_norm, umap_labels, f'{outdir}/figures', prefix='X__Original__PCA__UMAP')
            del adata_norm

    with time_code('Train PIANO model'):
        adata_train_list = [_[:, _.var['highly_variable']].copy() for _ in adata_train_list]  # Subset to genes used in training model
        if same_train_and_validation_data:
            adata_valid = adata_valid[:, adata_valid.var['highly_variable']].copy()  # Already a reference to the first training adata
        pianist = Composer(
            adata_train_list,
            categorical_covariate_keys = args.categorical_covariate_keys,
            continuous_covariate_keys = args.continuous_covariate_keys,
            n_top_genes=-1,
            hvg_batch_key=batch_key,
            max_epochs=args.max_epochs,
            run_name=run_name,
            outdir=outdir,
            memory_mode=memory_mode,
            num_workers=num_workers,
            adversarial=(args.adversarial == 'True'),
        )
        pianist.run_pipeline()
        pianist.save(f'{outdir}/pianist.pkl')

    with time_code('Validate PIANO model'):
        adata_valid.obsm['X__Original__PIANO'] = pianist.get_latent_representation(adata_valid)
        sc.pp.neighbors(adata_valid, n_neighbors=n_neighbors, n_pcs=pianist.model.latent_size, use_rep='X__Original__PIANO', random_state=random_state)
        sc.tl.umap(adata_valid, random_state=random_state)
        adata_valid.obsm['X__Original__PIANO__UMAP'] = adata_valid.obsm['X_umap']
        plot_umaps(adata_valid, umap_labels, f'{outdir}/figures', prefix='X__Original__PIANO__UMAP')
        del adata_valid.obsm['X_umap']
        print(adata_valid, flush=True)

    if args.plot_counterfactual:
        with time_code('Counterfactual analysis'):
            with time_code('Compute counterfactual expression'):
                adata_valid.layers['Counterfactual'] = pianist.get_counterfactual(None if same_train_and_validation_data else adata_valid)
                print("Counterfactual variance per gene:", np.var(adata_valid.layers['Counterfactual'], axis=0).mean())

            with time_code('Compute Counterfactual PCA UMAPs'):
                obs_columns_to_keep = np.unique(args.categorical_covariate_keys + args.continuous_covariate_keys + umap_labels)  # Avoid duplicating columns in .obs to avoid pandas bug
                adata_cf = ad.AnnData(
                    X=adata_valid.layers['Counterfactual'].copy() if args.save_counterfactual else adata_valid.layers['Counterfactual'],
                    obs=adata_valid.obs[obs_columns_to_keep].copy(),  # Copy only relevant columns for dataloader and umap plotting
                    var=pd.DataFrame(index=adata_valid.var_names.copy()),  # Do not modify reference to .var
                )
                if not args.save_counterfactual:
                    del adata_valid.layers['Counterfactual']
                adata_cf.obsm['X__Counterfactual__PIANO'] = pianist.get_latent_representation(adata_cf)
                adata_valid.obsm['X__Counterfactual__PIANO'] = adata_cf.obsm['X__Counterfactual__PIANO']
                sc.pp.normalize_total(adata_cf, target_sum=1e4)
                sc.pp.log1p(adata_cf)
                sc.pp.pca(adata_cf, n_comps=50, use_highly_variable=False)  # Avoid using hvg mask
                sc.pp.neighbors(adata_cf, n_neighbors=n_neighbors, n_pcs=n_pcs_pca, use_rep='X_pca', random_state=random_state)
                sc.tl.umap(adata_cf, random_state=random_state)
                if args.save_counterfactual or args.scib_benchmarking:
                    adata_valid.obsm['X__Counterfactual__PCA'] = adata_cf.obsm['X_pca']
                if args.save_counterfactual:
                    adata_valid.obsm['X__Counterfactual__PCA__UMAP'] = adata_cf.obsm['X_umap']
                plot_umaps(adata_cf, umap_labels, f'{outdir}/figures', prefix='X__Counterfactual__PCA__UMAP')

            with time_code('Compute Counterfactual PIANO UMAPs'):
                sc.pp.neighbors(adata_valid, n_neighbors=n_neighbors, n_pcs=pianist.model.latent_size, use_rep='X__Counterfactual__PIANO', random_state=random_state)
                sc.tl.umap(adata_valid, random_state=random_state)
                adata_valid.obsm['X__Counterfactual__PIANO__UMAP'] = adata_valid.obsm['X_umap']
                plot_umaps(adata_valid, umap_labels, f'{outdir}/figures', prefix='X__Counterfactual__PIANO__UMAP')
                if not args.save_counterfactual:
                    del adata_valid.obsm['X__Counterfactual__PIANO'], adata_valid.obsm['X__Counterfactual__PIANO__UMAP']
                del adata_valid.obsm['X_umap']

            with time_code('Compute Merged Original and Counterfactual PIANO UMAPs'):
                adata_valid.obs['Origin'] = 'Original'
                adata_cf.obs['Origin'] = 'Counterfactual'
                adata_merged = ad.AnnData(obs=pd.concat([
                    adata_valid.obs[umap_labels + ['Origin']],
                    adata_cf.obs[umap_labels + ['Origin']],
                ]))
                adata_merged.obsm['X__Merged__PIANO'] = np.vstack([adata_valid.obsm['X__Original__PIANO'], adata_cf.obsm['X__Counterfactual__PIANO']])
                sc.pp.neighbors(adata_merged, n_neighbors=n_neighbors, n_pcs=pianist.model.latent_size, use_rep='X__Merged__PIANO', random_state=random_state)
                sc.tl.umap(adata_merged, random_state=random_state)
                plot_umaps(adata_merged, umap_labels + ['Origin'], f'{outdir}/figures', prefix='X__Counterfactual_Merged__PIANO__UMAP')
                print(adata_merged, flush=True)
            del adata_cf, adata_merged

    if args.plot_reconstruction:
        with time_code('Reconstruction analysis'):
            adata_valid.layers['Reconstruction'] = pianist.get_counterfactual(None if same_train_and_validation_data else adata_valid, covariates=None)
            print("Reconstruction variance per gene:", np.var(adata_valid.layers['Reconstruction'], axis=0).mean())
            with time_code('Compute Reconstruction PCA UMAPs'):
                obs_columns_to_keep = np.unique(args.categorical_covariate_keys + args.continuous_covariate_keys + umap_labels)  # Avoid duplicating columns to keep in .obs to avoid pandas bug
                adata_cf = ad.AnnData(
                    X=adata_valid.layers['Reconstruction'].copy() if args.save_reconstruction else adata_valid.layers['Reconstruction'],
                    obs=adata_valid.obs[obs_columns_to_keep].copy(),  # Copy only relevant columns for dataloader and umap plotting
                    var=pd.DataFrame(index=adata_valid.var_names.copy()),  # Do not modify reference to .var
                )
                if not args.save_reconstruction:
                    del adata_valid.layers['Reconstruction']
                adata_cf.obsm['X__Reconstruction__PIANO'] = pianist.get_latent_representation(adata_cf)
                adata_valid.obsm['X__Reconstruction__PIANO'] = adata_cf.obsm['X__Reconstruction__PIANO']
                sc.pp.normalize_total(adata_cf, target_sum=1e4)
                sc.pp.log1p(adata_cf)
                sc.pp.pca(adata_cf, n_comps=50, use_highly_variable=False)  # Avoid using hvg mask
                sc.pp.neighbors(adata_cf, n_neighbors=n_neighbors, n_pcs=n_pcs_pca, use_rep='X_pca', random_state=random_state)
                sc.tl.umap(adata_cf, random_state=random_state)
                if args.save_reconstruction:
                    adata_valid.obsm['X__Reconstruction__PCA'] = adata_cf.obsm['X_pca']
                    adata_valid.obsm['X__Reconstruction__PCA__UMAP'] = adata_cf.obsm['X_umap']
                plot_umaps(adata_cf, umap_labels, f'{outdir}/figures', prefix='X__Reconstruction__PCA__UMAP')

            with time_code('Compute Reconstruction PIANO UMAPs'):
                sc.pp.neighbors(adata_valid, n_neighbors=n_neighbors, n_pcs=pianist.model.latent_size, use_rep='X__Reconstruction__PIANO', random_state=random_state)
                sc.tl.umap(adata_valid, random_state=random_state)
                adata_valid.obsm['X__Reconstruction__PIANO__UMAP'] = adata_valid.obsm['X_umap']
                plot_umaps(adata_valid, umap_labels, f'{outdir}/figures', prefix='X__Reconstruction__PIANO__UMAP')
                if not args.save_counterfactual:
                    del adata_valid.obsm['X__Reconstruction__PIANO'], adata_valid.obsm['X__Reconstruction__PIANO__UMAP']
                del adata_valid.obsm['X_umap']

            with time_code('Compute Merged Original and Reconstruction PIANO UMAPs'):
                adata_valid.obs['Origin'] = 'Original'
                adata_cf.obs['Origin'] = 'Reconstruction'
                adata_merged = ad.AnnData(obs=pd.concat([
                    adata_valid.obs[umap_labels + ['Origin']],
                    adata_cf.obs[umap_labels + ['Origin']],
                ]))
                adata_merged.obsm['X__Merged__PIANO'] = np.vstack([adata_valid.obsm['X__Original__PIANO'], adata_cf.obsm['X__Reconstruction__PIANO']])
                sc.pp.neighbors(adata_merged, n_neighbors=n_neighbors, n_pcs=pianist.model.latent_size, use_rep='X__Merged__PIANO', random_state=random_state)
                sc.tl.umap(adata_merged, random_state=random_state)
                plot_umaps(adata_merged, umap_labels + ['Origin'], f'{outdir}/figures', prefix='X__Reconstruction_Merged__PIANO__UMAP')
                print(adata_merged, flush=True)
            del adata_cf, adata_merged

    # Run scIB benchmarking
    if args.scib_benchmarking:
        with time_code('Integration Benchmarking'):
            bm = Benchmarker(
                adata_valid,
                batch_key=batch_key,
                label_key=args.celltype,
                embedding_obsm_keys=[_ for _ in ['X__Original__PCA', 'X__Original__PIANO', 'X__Counterfactual__PCA', 'X__Counterfactual__PIANO'] if _ in adata_valid.obsm],
                pre_integrated_embedding_obsm_key='X__Original__PCA',
                bio_conservation_metrics=BioConservation(
                    isolated_labels=False, nmi_ari_cluster_labels_leiden=True,
                    nmi_ari_cluster_labels_kmeans=False, silhouette_label=False, clisi_knn=False,
                ),
                batch_correction_metrics=BatchCorrection(
                    silhouette_batch=False, ilisi_knn=True, kbet_per_label=True,
                    graph_connectivity=False, pcr_comparison=False,
                ),
                n_jobs=-1,
            )
            bm.prepare()
            bm.benchmark()
            unscaled_bm_df = bm.get_results(min_max_scale=False).T
            unscaled_bm_df.to_csv(f'{outdir}/integration_results/bm_df.csv')
            print(unscaled_bm_df)
            del bm

    # Save integration results
    with time_code('Possibly saving Anndata'):
        if 'Origin' in adata_valid.obs:
            del adata_valid.obs['Origin']
        for k in ['neighbors', 'umap']:
            adata_valid.uns.pop(k, None)
        print(f"Final integrated data: {adata_valid}")
        if args.save_adata:
            adata_valid.write_h5ad(f'{outdir}/integration_results/adata_integrated.h5ad')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PIANO pipeline")
    parser.add_argument('--rach2', action='store_true', help="Piano Concerto No. 2 in C minor, Op. 18")

    # Run I/O parameters
    parser.add_argument("--version", type=str, default='0.0', help="Name of run")
    parser.add_argument("--adata_train_list", type=str, nargs='+', help="Path(s) to AnnData file(s)")
    parser.add_argument("--adata_valid", type=str, help="Path to AnnData file")
    parser.add_argument("--outdir", type=str, help="Path to output directory")

    # Model parameters
    parser.add_argument("--n_top_genes", type=int, default=4096, help="Number of highly variable genes")
    parser.add_argument("--categorical_covariate_keys", type=str, nargs='*', default=[], help="Categorical covariates to regress out")
    parser.add_argument("--continuous_covariate_keys", type=str, nargs='*', default=[], help="Continuous covariates to regress out")

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=200, help="Max number of training epochs")
    parser.add_argument("--adversarial", type=str, default='True', help="Use adversarial training (True/False). Default = True.")

    # Validation parameters
    parser.add_argument("--batch_key", type=str, help="Batch key for HVG selection")
    parser.add_argument("--geneset_path", type=str, default=None, help="Path to gene set to use instead of HVGs. Takes priority over HVGs.")
    parser.add_argument("--umap_labels", nargs='*', type=str, help="Colors for UMAPs")

    # Pipeline parameters
    parser.add_argument('--plot_unintegrated', action='store_true', help="Plot UMAPs of PCA of unintegrated gene expression")
    parser.add_argument('--plot_counterfactual', action='store_true', help="Plot UMAPs of PCA of counterfactual (batch-corrected) gene expression")
    parser.add_argument('--plot_reconstruction', action='store_true', help="Plot UMAPs of PCA of reconstruction of unintegrated gene expression")
    parser.add_argument('--n_pcs_pca', type=int, default=50, help="Number of PCs to use for PCA")
    parser.add_argument('--scib_benchmarking', action='store_true', help="Run integration benchmarking")
    parser.add_argument('--celltype', type=str, default='Group', help="Run integration benchmarking on cell type")

    # Script parameters
    parser.add_argument('--memory_profile', action='store_true', help="Run memory profiling")
    parser.add_argument('--save_adata', action='store_true', help="Save integrated adata")
    parser.add_argument('--save_original_pca', action='store_true', help="Save original (unintegrated) pca representations")
    parser.add_argument('--save_counterfactual', action='store_true', help="Save counterfactual (batch-corrected) counts")
    parser.add_argument('--save_reconstruction', action='store_true', help="Save VAE reconstruction counts")
    args = parser.parse_args()

    if args.rach2 or True:
        args.rach2 = 'Piano Concerto No. 2 in C minor, Op. 18'
        print("A Monsieur Sergei Rachmaninoff")
        print(vars(args))

    if args.memory_profile:
        main = profile(main)
    main(args)
