"""
scanpy's UMAP (sc.tl.umap) with umap-learn's spectral start, as piano's own copy, so the command line's CPU UMAPs do not
crash on large connected components (SCRUM-199).

umap-learn 0.5.12 starts the layout of each connected component from its spectral embedding, computed with ARPACK
(scipy's eigsh) below 2,000,000 cells with sqrt(n) Lanczos vectors: a work array of n x sqrt(n) values. From about
1.66M cells that passes 2^31 and ARPACK's 32-bit indices overflow, a segmentation fault without a Python error. The
copy below is umap-learn's, except that it uses lobpcg (umap-learn's own choice above 2M cells) wherever ARPACK would
overflow. Everything else is unchanged: for the same random_state the result equals sc.tl.umap's.

spectral_layout, multi_component_layout and _spectral_layout are copied from umap-learn 0.5.12 (umap/spectral.py), and
scanpy_umap follows scanpy 1.11.5's sc.tl.umap (scanpy/tools/_umap.py), both under the BSD 3-Clause License:

umap-learn: Copyright (c) 2017, Leland McInnes. All rights reserved.
scanpy: Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
  products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from warnings import warn

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from umap.spectral import component_layout
from umap.umap_ import find_ab_params, noisy_scale_coords, simplicial_set_embedding

# ARPACK indexes its work arrays with 32-bit integers
ARPACK_MAX_ENTRIES = 2**31 - 1
# umap-learn's own switch to lobpcg
UMAP_LOBPCG_MIN_VERTICES = 2_000_000


def eigensolver(n_vertices, dim):
    """
    The eigensolver for the spectral embedding of a connected graph of n_vertices in dim dimensions: umap-learn's choice
    (eigsh below 2,000,000 vertices, lobpcg from there on), except lobpcg wherever eigsh's Lanczos vectors would pass
    ARPACK's 32-bit indices (n_vertices x max(2 (dim + 1) + 1, sqrt(n_vertices)) entries, from ~1.66M vertices).
    """
    num_lanczos_vectors = max(2 * (dim + 1) + 1, int(np.sqrt(n_vertices)))
    if n_vertices >= UMAP_LOBPCG_MIN_VERTICES or n_vertices * num_lanczos_vectors > ARPACK_MAX_ENTRIES:
        return 'lobpcg'
    return 'eigsh'


def spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={}, tol=0.0, maxiter=0):
    """umap.spectral.spectral_layout: the spectral embedding of graph (the normalized Laplacian's eigenvectors)."""
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="random",
        tol=tol,
        maxiter=maxiter,
    )


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    tol=0.0,
    maxiter=0,
):
    """
    umap.spectral.multi_component_layout: place the connected components by a spectral embedding of their centroids,
    then embed each component spectrally at its place.
    """
    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
        else:
            component_embedding = _spectral_layout(
                data=None,
                graph=component_graph,
                dim=dim,
                random_state=random_state,
                metric=metric,
                metric_kwds=metric_kwds,
                init=init,
                tol=tol,
                maxiter=maxiter,
            )
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )

    return result


def _spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    tol=0.0,
    maxiter=0,
):
    """
    umap.spectral._spectral_layout with the eigensolver from eigensolver() (the only change), and without the method
    argument, which spectral_layout never passes.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
    L = I - D * graph * D
    if not scipy.sparse.issparse(L):
        L = np.asarray(L)

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    # The change: lobpcg wherever eigsh would overflow ARPACK, not only from 2,000,000 vertices
    method = eigensolver(L.shape[0], dim)

    try:
        if init == "random":
            X = gen.normal(size=(L.shape[0], k))
        else:
            raise ValueError(f"The init parameter must be 'random': {init} is invalid.")
        # For such a normalized Laplacian, the first eigenvector is always proportional to sqrt(degrees)
        X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

        if method == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=tol or 1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=maxiter or graph.shape[0] * 5,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error",
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    L,
                    np.asarray(X),
                    largest=False,
                    tol=tol or 1e-4,
                    maxiter=maxiter or 5 * graph.shape[0],
                )

        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except (scipy.sparse.linalg.ArpackError, UserWarning):
        warn(
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


def scanpy_umap(adata, min_dist=0.5, spread=1.0, n_components=2, maxiter=None, random_state=0):
    """
    sc.tl.umap on the neighbour graph of sc.pp.neighbors, with the spectral start above; writes .obsm['X_umap'] and
    .uns['umap'] as scanpy does. With the same random_state the result equals sc.tl.umap's.

    The spectral start is computed here and handed to umap-learn's simplicial_set_embedding as initial coordinates:
    on the graph it prunes, with the same random numbers in the same order, then scaled with the same noise, so the
    layout continues exactly as umap-learn's own spectral start would.
    """
    neighbors = adata.uns['neighbors']
    params = neighbors['params']
    if params.get('method', 'umap') != 'umap':
        raise ValueError(f"scanpy_umap needs the neighbour graph of sc.pp.neighbors(method='umap'), not {params['method']!r}")

    a, b = find_ab_params(spread, min_dist)
    adata.uns['umap'] = {'params': {'a': a, 'b': b}}
    if random_state != 0:
        adata.uns['umap']['params']['random_state'] = random_state
    random_state = check_random_state(random_state)

    # The representation the graph was built on, as scanpy chooses it: umap-learn uses it only to place the connected
    # components relative to each other
    x = adata.obsm[params['use_rep']] if params.get('use_rep') not in (None, 'X') else adata.X
    if params.get('n_pcs') is not None:
        x = x[:, :params['n_pcs']]

    # A copy, so the pruning below leaves .obsp['connectivities'] as it is (scanpy's own .tocoo() shares its values)
    graph = adata.obsp[neighbors.get('connectivities_key', 'connectivities')].tocoo(copy=True)
    n_epochs = (500 if graph.shape[0] <= 10000 else 200) if maxiter is None else maxiter

    # simplicial_set_embedding's pruning, applied to a copy for the spectral start
    pruned = graph.copy()
    pruned.sum_duplicates()
    default_epochs = 500 if pruned.shape[0] <= 10000 else 200
    pruned.data[pruned.data < (pruned.data.max() / float(n_epochs if n_epochs > 10 else default_epochs))] = 0.0
    pruned.eliminate_zeros()

    metric, metric_kwds = params.get('metric', 'euclidean'), params.get('metric_kwds', {})
    start = spectral_layout(x, pruned, n_components, random_state, metric=metric, metric_kwds=metric_kwds)
    start = noisy_scale_coords(start, random_state, max_coord=10, noise=0.0001)

    adata.obsm['X_umap'], _ = simplicial_set_embedding(
        data=x,
        graph=graph,
        n_components=n_components,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=5,
        n_epochs=n_epochs,
        init=start,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        densmap=False,
        densmap_kwds={},
        output_dens=False,
    )
