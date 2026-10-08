# Step 04 — Learn a Burgers time-step map and compare a physics-informed solve

This chapter adds two real CPU learning baselines to the experiment engine. The Fourier Neural Operator (FNO) learns a reusable map from one saved velocity field to the next. The physics-informed neural network (PINN) instead optimizes a solution for one held-out initial condition using the PDE, that initial field, and periodic boundary conditions.

The models answer different questions. FNO tests generalization across initial-condition families after amortized training. PINN tests whether a per-instance optimization can satisfy one initial-value problem. Their errors can be compared at the same physical coordinates, but their training costs and access to initial conditions must remain explicit.

The implementation is [ml.py](../../src/flowstate/ml.py); executable checks are in [test_ml.py](../../tests/test_ml.py). The original [FNO paper](https://arxiv.org/abs/2010.08895) motivates Fourier-space operator layers. The [physics-informed learning paper](https://arxiv.org/abs/1711.10561) motivates differentiating a neural solution to penalize its PDE residual. Flowstate implements small baselines around these ideas; it does not claim a novel architecture or full reproduction of either paper.

## 1. Start from a frozen trajectory dataset

The canonical dataset is a Zarr group with physical, unnormalized values:

```text
fields/u                 [trajectory, saved_time, x] float32
time                     [saved_time] float64
x                        [x] float64
splits/train              trajectory indices
splits/validation         trajectory indices
splits/test               trajectory indices
metadata.json             source runs, families, splits, normalization, provenance
manifest.json             SHA-256 hashes for the dataset artifacts
```

For example, twelve trajectories, eleven saved frames, and thirty-two spatial points give `fields/u.shape == (12, 11, 32)`. The frame interval is `saved_dt`; it is not necessarily the numerical solver's integration `dt`.

All trajectories must share the periodic Burgers equation, effective viscosity, grid, and saved physical times. The current FNO is not conditioned on viscosity or timestep. Mixing those quantities would ask the same input network to learn different evolution laws without telling it which law to use, so the dataset and model contracts reject that mixture.

The loader verifies the manifest and checks finite fields, increasing coordinates, complete disjoint splits, and family separation. Coordinate differences are treated as uniform within `rtol=1e-4, atol=1e-10`, matching the importer and allowing float32 source-coordinate rounding. Original coordinates are retained; this tolerance is stored in the model contract. It does not provide variable-timestep or irregular-grid training.

Entire initial-condition families are assigned to train, validation, or test before pairs are extracted. Related runs and exact duplicate initial fields must not be separated into training and test data. Only training trajectories determine the global normalization:

```text
z = (u - training_mean) / training_std
u = z * training_std + training_mean
```

Validation and test fields never fit the normalizer. The loader rechecks the stored statistics against the training split. Normalized windows are created separately inside each split, so adjacent frames of one trajectory cannot leak into another split.

Prepare the complete prototype environment once, then preserve it while running commands:

```sh
uv sync --locked --all-extras --group dev
uv run --no-sync flowstate dataset verify outputs/burgers-dataset
```

Here and below, `outputs/burgers-dataset` is an existing curated dataset from the data chapter. Use at least three distinct initial-condition families so all three splits are populated. The default demonstration uses more families, but remains a small example rather than a statistical benchmark.

## 2. Follow the FNO tensors through one forward pass

`_pairs()` turns each training trajectory into adjacent saved-frame pairs:

```text
input:  z(t0), z(t1), ..., z(tK-1)
target: z(t1), z(t2), ..., z(tK)
```

Pairs from training trajectories are flattened into a tensor `[number_of_pairs, x]`. A mini-batch is `[batch, x]`. `FNO1d` adds a channel axis and uses a pointwise convolution to lift the scalar field into `width` channels:

```text
[batch, x]
    → [batch, 1, x]
    → lift → [batch, width, x]
    → repeated Fourier/local blocks
    → projection → [batch, 1, x]
    → residual addition → [batch, x]
```

Each `SpectralConv1d` block does four operations:

1. `torch.fft.rfft` transforms each real spatial signal into its nonredundant complex Fourier coefficients.
2. Learned complex weights mix the input and output channels for the retained low modes. The weights have shape `[input_channel, output_channel, mode]`.
3. Unretained spectral-output modes are set to zero.
4. `torch.fft.irfft(..., n=grid_size)` returns the signal to the original spatial grid.

Both transforms use `norm="ortho"`. Supplying `n` to the inverse preserves odd as well as even grid sizes. The real FFT stores the independent half of a real signal's Hermitian spectrum; PyTorch differentiates through the FFT and complex weights. See the official [real FFT documentation](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html).

A parallel pointwise channel map is added to the spectral output before GELU activation. Therefore retaining a limited number of spectral modes does not globally erase all high-frequency information: the local path still processes the original latent field. There is no discontinuous coordinate ramp or zero padding at the periodic boundary. The network is translation equivariant on this periodic grid.

The final projection predicts a correction to the input field. Its last layer starts at zero, so epoch zero predicts persistence: the next field equals the current field. This provides an honest starting baseline and helps when the saved timestep is short. It is not an accuracy guarantee.

## 3. Inspect the training loop and validation decision

Run a small fit:

```sh
uv run --no-sync flowstate train-fno outputs/burgers-dataset outputs/fno-01 --epochs 20 --seed 0
```

Defaults are width 16, eight modes, three Fourier/local blocks, batch size 16, Adam learning rate 0.001, and twenty epochs. An epoch visits all training pairs once in a seeded shuffled order. For each mini-batch, the code:

1. Clears accumulated gradients with `optimizer.zero_grad()`.
2. Predicts the next normalized field and computes mean squared error against the training target.
3. Calls `loss.backward()` to propagate gradients through projection, pointwise paths, and Fourier weights.
4. Clips the total gradient norm at 10 and rejects non-finite loss or gradients.
5. Applies one Adam optimizer update.

After an epoch, evaluation without gradients computes training and validation MSE. The best validation model is retained separately from the last optimizer state. Epoch-zero persistence also participates in selection. If `best_epoch` is zero, training did not improve the validation criterion; the report says so instead of silently presenting a worse fit as progress.

The test split is evaluated after training and checkpoint selection, not used to select weights. Repeatedly changing settings after examining test results would still turn the test split into an informal validation set; a serious study needs a fresh final holdout or a preregistered comparison.

This implementation uses CPU float32 tensors, one PyTorch thread, a seeded CPU RNG, and deterministic algorithms. It restores the caller's prior RNG/thread settings on exit. Repeatability is tested within the same environment; it is not a promise of bitwise equivalence across PyTorch releases or machines.

The loader limits fields to sixteen million values. A training call permits at most 100,000 optimizer updates, with bounds on epochs, width, depth, modes, and batch size. These guards constrain the prototype; activation memory and runtime still depend on the chosen dimensions. Dataset export can stream trajectories, but the current learning loader loads its accepted dataset into memory.

## 4. Separate one-step skill from rollout stability

```sh
uv run --no-sync flowstate evaluate-fno outputs/burgers-dataset outputs/fno-01 --output outputs/fno-evaluation-01
```

`evaluate_fno()` uses the best-validation weights and produces two distinct predictions:

| Evaluation | Input at the next prediction | What it tests |
| --- | --- | --- |
| One step | The previous stored reference frame | Local map error when every input is corrected by the reference |
| Autoregressive rollout | The model's previous prediction, starting from the reference initial frame | Accumulation of prediction error without later reference corrections |
| One-step persistence | The previous stored reference frame, unchanged | Whether learning beats simply copying its input |
| Rollout persistence | The initial frame at every later time | Whether learned evolution beats a stationary trajectory |

All field errors are evaluated after converting predictions back to physical velocity. The report includes RMSE, relative L2 error, maximum error, mean-velocity error, energy error, and error per saved physical time. Relative L2 is null when the reference norm is zero. Non-finite predictions are explicitly flagged rather than encoded as JSON infinity.

The numerical reference has discretization error. A model's reported error measures agreement with that stored reference, not error against an exact solution. The report deliberately does not claim that FNO beats a spectral solver, that it is mesh-independent in this implementation, or that the reported inference time establishes a speedup. Training cost is reported separately, and useful speed comparisons need matched accuracy and controlled timing.

When `--output` is provided, evaluation writes `one_step.npy`, `rollout.npy`, `report.json`, and its own manifest to a new directory. The `.npy` arrays have shape `[evaluated_trajectory, future_saved_time, x]`; the initial frame is excluded because it is supplied as input. A model bundle's `verify` command applies to its checkpoint bundle, while the evaluation directory is a separate prediction artifact.

## 5. Understand exactly what a checkpoint can resume

A completed training call atomically publishes:

```text
outputs/fno-01/
  checkpoint.pt      last model + Adam state + RNG + epoch + best-validation model
  report.json        history, held-out evaluation, hashes, configuration, provenance
  manifest.json      artifact hashes
```

The dataset manifest hash identifies the exact dataset, including its split and preprocessing metadata. The report also records the checkpoint hash, source/environment provenance, training configuration, and an optional parent checkpoint hash. The manifest detects accidental changes; it is unsigned and does not make external edits impossible.

Check the bundle and continue into a different directory:

```sh
uv run --no-sync flowstate model verify outputs/fno-01
uv run --no-sync flowstate train-fno outputs/burgers-dataset outputs/fno-02 --epochs 10 --seed 0 --resume outputs/fno-01
```

Here `--epochs 10` means **ten additional epochs**. Resume restores the last model, optimizer moments, CPU RNG, and epoch counter; it does not restart optimization from the best-validation weights. Hyperparameters, learning implementation, PyTorch version, dataset hash, and grid/time/viscosity/normalization contract must match. Pass the same nondefault flags if the original run used them.

Checkpoint loading uses `torch.load(..., map_location="cpu", weights_only=True)` with tensor/state dictionaries rather than pickled model objects. The artifact manifest is verified before loading. See [PyTorch's loader documentation](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

Publication occurs at the end of a training call. This is resumption from a completed checkpoint, not recovery of an unfinished epoch after a crash. Existing model directories are never overwritten, and output cannot be written inside the frozen dataset. Run shorter increments when checkpoint frequency matters.

## 6. Derive the PINN loss in physical coordinates

```sh
uv run --no-sync flowstate train-pinn outputs/burgers-dataset outputs/pinn-01 --epochs 200 --seed 0
```

By default, this selects the first test trajectory; `--trajectory-index` selects another index that must also belong to the test split. The model is a tanh multilayer perceptron `u_theta(t,x)`. It receives physical time and position, scales those coordinates inside its differentiable forward pass, and converts its scalar output to physical velocity using training-split normalization.

The derivatives are with respect to the original physical inputs. Autograd therefore includes the factors from input and output scaling:

```text
u_t = ∂u_theta/∂t
u_x = ∂u_theta/∂x
u_xx = ∂²u_theta/∂x²
r = u_t + u_theta * u_x - viscosity * u_xx
```

`burgers_residual()` constructs this expression with `create_graph=True`, so a subsequent backward pass differentiates the residual loss with respect to model parameters. A constant spatial derivative has an exact zero second derivative; the helper handles that case without confusing an unused coordinate with an autograd failure.

Each epoch samples new seeded space-time collocation points. The objective contains three terms:

```text
10 × initial-condition MSE
 + PDE-residual MSE
 + periodic-boundary MSE for both u and u_x
```

Initial and boundary velocity errors are divided by training standard deviation. The PDE residual is scaled by `training_std / time_duration`; the boundary derivative difference is scaled by `domain_length / training_std`. These dimensionless scalings and the initial-condition weight are fixed choices, not tuned on the held-out interior trajectory.

Only the selected trajectory's first field supplies supervised targets. Interior values from that trajectory never appear in `_pinn_loss()`. Boundary losses compare the network's two periodic endpoints to each other, and residual losses compare the PDE expression to zero. The test interior is used only after optimization to measure trajectory error. The checkpoint is the last epoch, with no interior-reference-based selection.

The PINN report contains the initial, residual, and boundary loss histories; a physical residual RMS at fresh collocation points; errors at the same future reference times; a persistence comparison; and the explicit label that this is per-instance physics optimization. `predictions.npy` contains `[saved_time, x]`, including the predicted initial frame. A low residual alone does not establish an accurate solution: inspect the initial/boundary fit and trajectory errors together.

PINN also supports `--resume` into a new directory with the same hyperparameters and selected trajectory. The optimizer and collocation RNG are restored. Its call budget is capped at two million sampled residual points, and short demonstration training may not converge.

## 7. Reproduce the checks and interpret the result

```sh
uv run --no-sync pytest tests/test_ml.py -q
```

The tests check retained Fourier modes and complex-weight gradients; actual learning of a smooth decay map; exact final-state agreement between uninterrupted and resumed FNO/PINN training; matching evaluation and persistence calculations; manifest tamper rejection; float32 coordinate rounding; a known Burgers residual; physical derivative scaling; and invariance of PINN losses and gradients when held-out interior targets are changed.

Use the resulting artifacts to ask a concrete next question: which saved timestep, viscosity, or initial-condition family produces the largest rollout error, and does a finer numerical reference change that conclusion? Each changed numerical problem needs a corresponding compatible dataset/model contract. This implementation supplies evidence for that investigation; larger FNO benchmarks, parameter-conditioned operators, uncertainty calibration, and broad PINN comparisons require additional experiments.
