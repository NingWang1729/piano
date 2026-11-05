# Changelog

All notable changes to PIANO are listed here, newest first, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the package version on PyPI.

## [Unreleased]

### Added

- Checkpoints at chosen epochs: `Composer(checkpoint_epochs=[50, 100])` (command line: `--checkpoint_epochs 50 100`, in
  both scripts) saves the weights after exactly 50 and 100 trained epochs to `checkpoints/model_epochs=50.pt` and
  `checkpoints/model_epochs=100.pt`, in PIANO and scVI mode, and the best epoch's weights to `checkpoints/model_best.pt`
  (its number of epochs in `best_epoch`): in PIANO mode the final model's epoch, by the early stopping metric; in scVI
  mode the epoch with the lowest validation ELBO, then scored every epoch. Values must be integers from 1 to
  `max_epochs`, each once (else a `ValueError` at creation). The KL weight warms up over `n_annealing_epochs` (by
  default 200, or 400 in scVI mode), so earlier checkpoints are taken while it still rises. A listed epoch that training does not reach (early stopping, or a lowered `max_epochs`) warns, and in scVI mode a
  validation ELBO that is never finite skips `model_best.pt` with a warning. The checkpoints do not change training.
- The last epoch's weights are always saved to `checkpoints/model_last.pt` (the epoch early stopping stopped at, if it
  did). In PIANO mode the final model is the best epoch, so this can differ from `model_checkpoint.pt`. Loading it and
  training again fine-tunes from those weights (the KL warm-up and the optimizer start again); resuming training
  exactly is not supported yet. Train on into a new `outdir` (README, Checkpoints), as the file names are fixed.
- A Composer pickled by an earlier version (`Composer.save`) lacks these settings, so it cannot be trained again with
  this version: pickles carry no compatibility promise between versions. Its weights (`checkpoints/*.pt`) still load
  with `load_model`.
- Continuous covariates in scVI mode: `Composer(mode='scvi', continuous_covariate_keys=[...])`, as scvi-tools'
  `SCVI` with `continuous_covariate_keys`. Raw values (not rescaled, as scVI) are appended to the latent at the
  decoder's input; the encoder does not see them. Weights load from and match a scvi-tools VAE with continuous
  covariates. Sparse continuous covariates still raise `NotImplementedError` in scVI mode.
- `--compile_model` (`True`/`False`, default `True`) in both command-line scripts, to turn compilation off.

### Changed

- `compile_model=True` (the default) now compiles the training step on the CPU too, not only on a CUDA GPU, which
  trains faster after a one-off compilation in the first epoch. On the CPU this needs a C++ compiler (`$CXX`, else
  `g++`, or `clang++` on macOS); without one the Composer raises a `RuntimeError` when it is created. Pass
  `compile_model=False` (`--compile_model False` on the command line) for the old behaviour. A GPU memory mode falling
  back to the CPU no longer turns compilation off.

### Fixed

- The command line's CPU UMAPs crashed with a segmentation fault (no Python error) when the neighbour graph had one
  connected component of about 1.66M-2M cells: umap-learn's spectral start overflows ARPACK's 32-bit indices there.
  The command line now uses piano's copy of `sc.tl.umap` (`piano.utils.scanpy_umap`), which gives the same UMAP and
  uses lobpcg for such components. The GPU path (rapids-singlecell) is unchanged.
- The command line's CPU UMAPs failed on 8,192 cells or more with pynndescent 0.7.0 (`AttributeError: 'csr_array'
  object has no attribute 'getnnz'`, from scanpy 1.11): pynndescent is now pinned below 0.7.

## [0.2.0] - 2026-11-03

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
- `import piano` no longer needs triton, which only the SparseGPU memory mode uses and which exists only on Linux (it
  comes with torch's CUDA builds there): piano now imports and trains in the other memory modes on macOS and Windows.
  `SparseTritonMatrix` loads on first use, and SparseGPU raises an ImportError naming triton if it is missing.
- pandas 3 is not supported yet: piano 0.2.0 requires pandas below 3.
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

### Fixed

- Categorical covariates work with pandas 3: encoding their labels no longer passes a list to `pd.factorize`, which
  pandas 3 rejects (it raised TypeError when creating the Composer) and pandas 2 warned about. Labels and codes are
  unchanged. Found and fixed by Wen-Jou Chang.

### Security

- `Composer.load` only unpickles the classes a Composer saved by `Composer.save` holds (piano's models and datasets,
  their torch layers, and numpy and pandas objects). A file naming any other class or function raises
  `pickle.UnpicklingError` before that code can run, and tensors load with `torch.load(weights_only=True)`. This is not
  a sandbox: only load files you trust.
