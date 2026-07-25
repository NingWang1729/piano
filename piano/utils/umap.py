import os
import gc
import multiprocessing

import anndata as ad
import faiss
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.sparse import coo_matrix
from umap.layouts import optimize_layout_euclidean
from umap.umap_ import compute_membership_strengths, find_ab_params, general_simplicial_set_union, make_epochs_per_sample, smooth_knn_dist
from piano.utils.timer import time_code


def faiss_knn(
    X,
    k=30,
    nlist=2**17, #131,072 OR use 2^16 = 65536
    nprobe=256,
    train_size=2**23, #8,388,608 OR use 2^22 = 4,194,304
    vector_addition_batch_size=2**20,  # 1,048,576
    probe_search_batch_size=2**19,  # 524,288
    additional_k_buffer_size=30,
    random_state=0,
):
    X = np.asarray(X, dtype=np.float32, order="C")
    n, d = X.shape

    # ----------------------------
    # Train IVF coarse quantizer
    # ----------------------------
    rng = np.random.default_rng(random_state)
    train_size = min(train_size, n)
    train_idx = rng.choice(n, size=train_size, replace=False, shuffle=False)
    train_X = X[train_idx]

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(train_X)
    del train_X, train_idx
    gc.collect()

    # ----------------------------
    # Add all vectors
    # ----------------------------
    batch = vector_addition_batch_size
    for i in range(0, n, batch):
        index.add(X[i:i+batch])

    # Search multiple inverted lists
    index.nprobe = nprobe
    batch = probe_search_batch_size
    indices = np.empty((n, k), np.int32)
    distances = np.empty((n, k), np.float32)

    for i in range(0, n, batch):
        end = min(i + batch, n)
        D, I = index.search(
            X[i:end],
            k + additional_k_buffer_size,
        )
        rows = np.arange(i, end, dtype=np.int64)
        if np.all(I[:, 0] == rows):
            indices[i:end] = I[:, 1:k+1]
            distances[i:end] = D[:, 1:k+1]
        else:
            out_I = np.empty((len(I), k), np.int32)
            out_D = np.empty((len(I), k), np.float32)
            for j in range(len(I)):
                # keep = I[j] != rows[j]
                keep = (I[j] != rows[j]) & (I[j] >= 0)
                neighbors = I[j][keep]
                if len(neighbors) < k:
                    print(f"Warning: Only found {len(neighbors)} neighbors; increase nprobe or additional_k_buffer_size.")
                out_I[j] = neighbors[:k]
                out_D[j] = D[j][keep][:k]
            indices[i:end] = out_I
            distances[i:end] = out_D
    del index
    gc.collect()

    return indices, distances

def faiss_knn_to_umap_graph(
    knn_indices,
    knn_dists,
    batch_size=524_288,
    symmetrize=True,
):
    """
    Convert FAISS kNN output into a UMAP-style fuzzy graph.

    Parameters
    ----------
    knn_indices : np.ndarray
        Shape (n_cells, n_neighbors), integer neighbor indices.

    knn_dists : np.ndarray
        Shape (n_cells, n_neighbors), float32 distances.

    batch_size : int
        Number of cells processed at once.

    symmetrize : bool
        Apply UMAP fuzzy union:
            w_ij + w_ji - w_ij*w_ji

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        Weighted graph.
    """

    if np.any(knn_indices < 0):
        print(f"Warning: FAISS returned invalid neighbor indices: {np.mean(knn_indices < 0)}")

    n, k = knn_indices.shape
    rows = np.empty(n * k, np.int32)
    cols = np.empty(n * k, np.int32)
    vals = np.empty(n * k, np.float32)

    offset = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        knn_indices_batch = knn_indices[start:end]
        knn_dists_batch = knn_dists[start:end]

        # Calculate batch
        sigmas_batch, rhos_batch = smooth_knn_dist(knn_dists_batch, k)
        rows_batch, cols_batch, vals_batch, _ = compute_membership_strengths(knn_indices_batch, knn_dists_batch, sigmas_batch, rhos_batch, return_dists=False)  # dists is returned as None
        m = len(rows_batch)

        # Aggregate batch
        rows[offset:offset+m] = rows_batch.astype(np.int32, copy=False) + start
        cols[offset:offset+m] = cols_batch.astype(np.int32, copy=False)
        vals[offset:offset+m] = vals_batch.astype(np.float32, copy=False)
        del knn_indices_batch, knn_dists_batch, sigmas_batch, rhos_batch, rows_batch, cols_batch, vals_batch
        offset += m

    graph = coo_matrix(
        (vals[:offset], (rows[:offset], cols[:offset])),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    del rows, cols, vals

    if symmetrize:
        graph_T = graph.T
        graph = general_simplicial_set_union(graph, graph_T)
        del graph_T

    del knn_indices, knn_dists
    gc.collect()
    
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(graph, directed=False)
    print(f"Neighborhood graph contains {n_components} components")

    return graph

def pca_init(X, batch_size=2**20):
    n, d = X.shape
    mean = X.mean(axis=0)
    C = np.zeros((d, d), dtype=np.float64)

    # Compute covariance
    for i in range(0, n, batch_size):
        Xi = X[i:i+batch_size].astype(np.float64)
        Xi -= mean
        C += Xi.T @ Xi
    C /= (n - 1)

    # Get top two PCs
    _, components = np.linalg.eigh(C)
    components = components[:, -2:]
    embedding = np.empty((n, 2), dtype=np.float32)

    # Projection into top two PCs
    for i in range(0, n, batch_size):
        Xi = X[i:i+batch_size] - mean
        embedding[i:i+batch_size] = Xi @ components

    return embedding

def optimize_embedding(
    X,
    graph,
    n_epochs=500,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5,
    parallel=True,
    verbose=True,
    random_state=0,
):
    n_vertices = len(X)
    a, b = find_ab_params(
        spread=1.0,
        min_dist=0.1,
    )
    # embedding = random_init(n_vertices, random_state)
    with time_code('PCA init'):
        embedding = pca_init(X)

    head, tail = graph.nonzero()
    epochs_per_sample = make_epochs_per_sample(
        graph.data,
        n_epochs,
    )
    rng_state = np.random.RandomState(random_state).randint(
        0, 2**31 - 1,
        size=3,
        dtype=np.int64,
    )
    embedding = optimize_layout_euclidean(
        head_embedding=embedding,
        tail_embedding=embedding,
        head=head.astype(np.int32),
        tail=tail.astype(np.int32),
        n_epochs=n_epochs,
        n_vertices=n_vertices,
        epochs_per_sample=epochs_per_sample,
        a=a,
        b=b,
        rng_state=rng_state,
        gamma=gamma,
        initial_alpha=initial_alpha,
        negative_sample_rate=negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
    )

    return embedding

def faiss_umap(
    X,

    # FAISS KNN
    n_neighbors=30,
    nlist=2**16, # 65,536 OR use 2^17 = 131,072
    nprobe=1024,
    train_size=2**23, #8,388,608 OR use 2^22 = 4,194,304
    vector_addition_batch_size=2**20,  # 1,048,576
    probe_search_batch_size=2**19,  # 524,288
    additional_k_buffer_size=30,

    # KNN to Graph
    knn_to_graph_batch_size=2**19,  # 524,288
    symmetrize=True,

    # Embedding
    n_epochs=200,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5,
    parallel=True,
    verbose=True,
    random_state=0,
    num_threads=16,
):
    n_cores = min(num_threads, multiprocessing.cpu_count())
    if n_cores < num_threads:
        print(f"Warning: num_threads {num_threads} is more than available cores {n_cores}, setting to {n_cores}")
        num_threads = n_cores
    os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
    faiss.omp_set_num_threads(num_threads)

    with time_code('FAISS'):
        knn_indices, knn_dists = faiss_knn(
            X, 
            k=n_neighbors,
            nlist=nlist, nprobe=nprobe, train_size=train_size,
            vector_addition_batch_size=vector_addition_batch_size,
            probe_search_batch_size=probe_search_batch_size,
            additional_k_buffer_size=additional_k_buffer_size,
            random_state=random_state,
        )
    
    with time_code('GRAPH'):
        graph = faiss_knn_to_umap_graph(
            knn_indices,
            knn_dists,
            batch_size=knn_to_graph_batch_size,
            symmetrize=symmetrize,
        )
        del knn_indices, knn_dists

    with time_code('EMBEDDING'):
        embedding = optimize_embedding(
            X, graph, n_epochs=n_epochs,
            gamma=gamma, initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            parallel=parallel, verbose=verbose,
            random_state=random_state,
        )

    return embedding

def plot_umaps(adata, umap_labels, outdir, umap_key='X_umap', show_interactive=False, x_outlier_pct_threshold=None, y_outlier_pct_threshold=None):
    # Helper function for visualization. Included here in full to enable easy user modifications
    umap_labels = list(dict.fromkeys(umap_labels))
    adata_perm = ad.AnnData(obs=adata.obs[umap_labels])
    adata_perm.obsm['X_umap'] = adata.obsm[umap_key]
    adata_perm = adata_perm[np.random.permutation(np.arange(adata.shape[0]))].copy()  # Expensive, but avoids N x N sparse indexing cost

    if x_outlier_pct_threshold is not None:
        x = adata_perm.obsm['X_umap'][:, 0]
        xmin, xmax = np.percentile(x, [x_outlier_pct_threshold, 100 - x_outlier_pct_threshold])
        xpad = 0.05 * (xmax - xmin)
        xmin -= xpad
        xmax += xpad
    if y_outlier_pct_threshold is not None:
        y = adata_perm.obsm['X_umap'][:, 1]
        ymin, ymax = np.percentile(y, [y_outlier_pct_threshold, 100 - y_outlier_pct_threshold])
        ypad = 0.05 * (ymax - ymin)
        ymin -= ypad
        ymax += ypad
    if x_outlier_pct_threshold is not None and y_outlier_pct_threshold is not None:
        outliers = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
        print(f"{outliers.sum():,} / {len(outliers):,} cells ({outliers.mean()*100:.3f}%) outside X and Y outlier thresholds")
    elif x_outlier_pct_threshold is not None:
        outliers = (x < xmin) | (x > xmax)
        print(f"{outliers.sum():,} / {len(outliers):,} cells ({outliers.mean()*100:.3f}%) outside X outlier thresholds")
    elif y_outlier_pct_threshold is not None:
        outliers = (y < ymin) | (y > ymax)
        print(f"{outliers.sum():,} / {len(outliers):,} cells ({outliers.mean()*100:.3f}%) outside Y outlier thresholds")

    os.makedirs(outdir, exist_ok=True)
    for umap_label in umap_labels:
        fig = sc.pl.umap(adata_perm, color=umap_label, return_fig=True)
        # Remove outliers from plot:
        if x_outlier_pct_threshold is not None:
            fig.axes[0].set_xlim((xmin, xmax))
        if y_outlier_pct_threshold is not None:
            fig.axes[0].set_ylim((ymin, ymax))
        legend = fig.axes[0].get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((0.5, -0.1))
            legend.set_loc('upper center')
        fig.savefig(f'{outdir}/{umap_key}__{umap_label}.png', bbox_inches='tight')
        if show_interactive:
            plt.show()
        plt.close(fig)
