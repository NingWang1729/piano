# Changelog

All notable changes to PIANO are listed here, newest first, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the package version on PyPI.

## [0.2.0] - Unreleased

### Changed

New training defaults, in the Composer and, where they have the option, both command-line scripts. Pass the old value
to get the old behaviour back:

- Model: 4 hidden layers of 512 units (was 3 x 256): `n_layers=3, n_hidden=256`.
- Batch size 512 (was 128): `batch_size=128`. Data with fewer than 512 training cells needs a smaller `batch_size` or
  `drop_last=False`.
- TF32 matrix multiplications on Ampere and newer GPUs (`matmul_precision='high'`; the command-line scripts already used
  TF32): `matmul_precision='highest'` for full FP32.
- No dropout in the decoder (`decoder_dropout_rate=0`): `decoder_dropout_rate=None` uses `dropout_rate` there, as
  before.
- Constant adversary weight 0.5 (was 1): `min_adv_weight=1, max_adv_weight=1`.
- KL weight annealed to 0.5 over a fixed 200 epochs, then held (was 0.25 over `max_epochs`):
  `max_kld_weight=0.25` and `n_annealing_epochs` equal to `max_epochs`.
- No early stopping (was on in the Composer): `early_stopping=True`.
- 50 training epochs (was 200): `max_epochs=200`.
- Learning rate: still 2e-4, now documented (the README listed 5e-4 by mistake).

Other changes:

- PIANO's ZINB likelihood (`distribution='zinb'`) is computed in plain torch instead of with pyro's
  `ZeroInflatedNegativeBinomial`: the same loss and gradients, but compiled ZINB training (`compile_model=True`) no longer
  splits into many graphs and runs about 25-30% faster. Compiled ZINB results may differ from before by float noise.
- Dependencies: `h5py`, `numba`, `statsmodels`, `tqdm` and `umap-learn`, which piano imports, are now declared; they
  used to arrive only because anndata and scanpy depend on them.
- The command line (`python -m piano`) and `plot_umaps` no longer need faiss, so they work on a plain install; only
  `faiss_knn`, `faiss_umap` and `FAISS_KNN` need it (the `misc` extra); `faiss_knn` and `faiss_umap` say so if it is
  missing.
- Packaging: `pyproject.toml` alone configures the build (setuptools 77 or newer), and the wheel holds only the `piano`
  package.

### Added

- `decoder_dropout_rate`: dropout of the decoder layers, separate from `dropout_rate`, which applies to the encoder
  only.
- `matmul_precision` (`'high'` or `'highest'`): the float32 matmul precision of training and inference, set only while
  the Composer runs the model; the command-line scripts no longer change it process-wide.
- scVI mode, `Composer(mode='scvi', batch_key=...)`: scvi-tools' model and training recipe with an NB or ZINB likelihood
  (`distribution='zinb'`), on PIANO's faster training path. Its defaults follow scvi-tools except a 4 x 512 network,
  batch size 512 and TF32; see "scVI mode" in the README.

### Removed

- Support for Composers pickled with PIANO 0.1.x: they lack settings added in 0.2.0 and fail when used. Create and train
  them again with 0.2.0.
- The `pyro-ppl` dependency: piano no longer uses pyro. The tests still compare against it; it is in the `misc` extra.
- `setup.py`: its metadata was already overridden by `pyproject.toml`.

### Security

- `Composer.load` only unpickles the classes a Composer saved by `Composer.save` holds (piano's models and datasets,
  their torch layers, and numpy and pandas objects). A file naming any other class or function raises
  `pickle.UnpicklingError` before that code can run, and tensors load with `torch.load(weights_only=True)`. This is not
  a sandbox: only load files you trust.
