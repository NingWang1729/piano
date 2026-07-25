import faiss
import numpy as np
from piano.utils.timer import time_code


class FAISS_KNN:
    def __init__(
        self,
        k=15,
        nlist=2**16,
        nprobe=256,
        max_per_class=6400,
        search_batch_size=2**19,
        metric=faiss.METRIC_L2,
        num_threads=None,
        random_state=42,
    ):
        self.k = k
        self.nlist = nlist
        self.nprobe = nprobe
        self.max_per_class = max_per_class
        self.search_batch_size = search_batch_size
        self.metric = metric
        self.num_threads = num_threads
        self.random_state = random_state
        self.index = None
        self.labels = None
        self.classes_ = None
        self.label_ids = None
        self.n_classes = None

    def _stratified_subset(self, X, y):
        """
        Sample up to max_per_class examples per label.
        """
        rng = np.random.default_rng(self.random_state)
        indices = []
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            n = min(self.max_per_class, len(idx))
            indices.append(
                rng.choice(idx, size=n, replace=False)
            )
        indices = np.concatenate(indices)

        return X[indices], y[indices]


    def fit(self, X, y):
        """
        Train FAISS IVF index.

        Parameters
        ----------
        X : array-like, float32
            Training embeddings.
        y : array-like
            Class labels.
        """

        if self.num_threads is not None:
            faiss.omp_set_num_threads(self.num_threads)

        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y)

        with time_code("Prepare training subset"):
            X_subset, y_subset = self._stratified_subset(X, y)
            self.classes_, y_encoded = np.unique(y_subset, return_inverse=True)
            self.n_classes = len(self.classes_)
            print(f"FAISS training set: {X_subset.shape}")

        with time_code("Build FAISS index"):
            d = X.shape[1]
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, self.metric)
            self.index.train(X_subset)
            self.index.add(X_subset)
            self.labels = y_encoded

        return self

    def predict(self, X):
        """
        Predict labels using weighted kNN voting.
        """

        X = np.asarray(X, dtype=np.float32, order="C")
        self.index.nprobe = self.nprobe
        predictions = np.empty(len(X), dtype=np.int32)
        rows = np.repeat(np.arange(self.search_batch_size), self.k)
        with time_code("FAISS prediction"):
            for start in range(0, len(X), self.search_batch_size):
                end = min(start + self.search_batch_size, len(X))
                batch_size = end - start
                D, I = self.index.search(X[start:end], self.k)
                neighbor_labels = self.labels[I]
                weights = 1.0 / (D + 1e-8)
                scores = np.zeros((batch_size, self.n_classes), dtype=np.float32)
                np.add.at(scores, (rows[:batch_size * self.k], neighbor_labels.ravel()), weights.ravel())
                predictions[start:end] = scores.argmax(axis=1)

        return self.classes_[predictions]
