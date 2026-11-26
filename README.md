# ERLPlace

Official code for **“An Evolutionary Reinforcement Learning Framework for Macro Placement Optimization.”**

ERLPlace is a hybrid evolutionary reinforcement learning framework for macro placement. It combines the global exploration capability of an evolutionary algorithm (EA) with the local exploitation capability of proximal policy optimization (PPO). A population of actor-critic agents shares experience through a dual-component replay buffer, while element-wise crossover and rollback are used to improve population diversity and training stability.

## Overview

The framework contains three main components:

- **Global search:** non-elite agents inherit parameters from experienced agents through element-wise crossover.
- **Local search:** PPO updates the actor and critic networks using a shared replay buffer.
- **Evaluation and rollback:** each agent is evaluated on the macro placement environment, and agents with degraded performance after crossover can be restored to their previous parameters.

The implementation supports the ISPD2005 macro placement benchmarks and follows the network design used by EfficientPlace.

## Repository Structure

```text
ERLPlace/
├── benchmarks/             # ISPD2005 benchmark instances
│   ├── adaptec1/
│   ├── adaptec2/
│   ├── adaptec3/
│   ├── adaptec4/
│   ├── bigblue1/
│   └── bigblue3/
├── configs/                # Per-benchmark training configurations
├── envs/                   # Placement environment and benchmark parser
│   ├── __init__.py
│   └── PlaceDB.py
├── models/                 # Actor, critic, replay buffer, and PPO agent
│   ├── __init__.py
│   ├── Actor.py
│   ├── Buffer.py
│   ├── Critic.py
│   └── Layer.py
├── trainer/                # ERL training loop and solution pool
│   ├── __init__.py
│   └── solution_pool.py
├── config.yaml             # Global runtime configuration
├── main.py                 # Training entry point
├── LICENSE                 # GNU General Public License v3.0
└── README.md
```

## Requirements

The code is designed for Python 3.14 and has been tested with Python 3.14.10 and PyTorch 2.15.0. CUDA is recommended for full benchmark training, while the code also supports CPU execution when CUDA is unavailable.

An example Conda environment is shown below:

```bash
conda create -n erlplace python=3.14 -y
conda activate erlplace

pip install \
    torch==2.15.0 \
    numpy \
    pyyaml \
    pillow \
    tensorboardX \
    tensorboard \
    treelib
```

For CUDA training, install the PyTorch build that matches your CUDA driver from the official PyTorch package index. For CPU-only machines, install the corresponding CPU build.

## Benchmarks

The repository includes the following ISPD2005 benchmark instances:

- `adaptec1`
- `adaptec2`
- `adaptec3`
- `adaptec4`
- `bigblue1`
- `bigblue3`

Each benchmark directory contains the standard `.aux`, `.nets`, `.nodes`, `.pl`, `.scl`, and `.wts` files. The benchmark parser is implemented in `envs/PlaceDB.py`.

The `bigblue3.nets` file exceeds GitHub's 100 MB per-file limit. It is therefore stored as two byte-contiguous parts:

- `benchmarks/bigblue3/bigblue3.nets.1` — 78,736,123 bytes
- `benchmarks/bigblue3/bigblue3.nets.2` — 77,044,618 bytes

The second part starts at a complete `NetDegree` record. Reconstruct the original file before running the Bigblue3 benchmark.

### Restoring `bigblue3.nets`

On Linux or macOS:

```bash
cd benchmarks/bigblue3
cat bigblue3.nets.1 bigblue3.nets.2 > bigblue3.nets
```

On Windows:

```bat
cd benchmarks\bigblue3
cmd /c copy /b bigblue3.nets.1+bigblue3.nets.2 bigblue3.nets
```

The restored file should report `NumNets : 1123170` and `NumPins : 3833218` in its header. Training automatically reads `bigblue3.nets`, so the reconstructed file must be present before launching a Bigblue3 run. Do not commit the reconstructed file; keep it local because it exceeds GitHub's file-size limit. Please make sure there is no blank line between two files.

## Training

Run all commands from the repository root because benchmark, configuration, and output paths are resolved relative to the current working directory.

To train with the default arguments:

```bash
python main.py
```

To specify the random seed, GPU, benchmark, output folder, and logging level:

```bash
python main.py \
    -s 42 \
    -g 0 \
    -b adaptec1 \
    -f erlplace_adaptec1 \
    -l info
```

### Command-Line Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `-s`, `--seed` | Random seed | `42` |
| `-g`, `--gpu` | CUDA device exposed through `CUDA_VISIBLE_DEVICES` | `0` |
| `-b`, `--benchmark` | Benchmark name; must match a directory under `benchmarks/` and a YAML file under `configs/` | `adaptec1` |
| `-f`, `--folder` | Output folder for logs, models, placements, and TensorBoard events | `VSCode_Debug` |
| `-l`, `--log` | Logging level: `debug`, `info`, `warning`, `error`, or `critical` | `info` |

For example, to train on Bigblue3 using GPU 1:

```bash
python main.py \
    -s 7 \
    -g 1 \
    -b bigblue3 \
    -f erlplace_bigblue3 \
    -l info
```

## Configuration

Global runtime settings are defined in `config.yaml`, and benchmark-specific settings are defined in `configs/<benchmark>.yaml`.

Important benchmark parameters include:

| Parameter | Description |
| --- | --- |
| `number_of_macros_to_place` | Number of macros selected for policy-driven placement |
| `grid` | Resolution of the placement canvas |
| `max_episodes` | Maximum number of environment interactions |
| `population_size` | Number of actor-critic agents in the population |
| `elite_count` | Number of best-performing agents treated as elites |
| `buffer_capacity` | Capacity assigned to each part of the dual replay buffer |
| `layer_crossover_begin` | Minimum iteration before crossover becomes eligible; the actual trigger also requires alignment with the crossover interval |
| `layer_crossover_interval` | Crossover interval in iterations |
| `layer_crossover_prob` | Per-parameter crossover probability |
| `layer_rollback_threshold` | Performance threshold used to decide whether to roll back an agent |
| `ppo_epoch` | Number of PPO optimization epochs |
| `ppo_batch_size` | PPO minibatch size |
| `actor_learning_rate` | Actor learning rate |
| `critic_learning_rate` | Critic learning rate |
| `clip_epsilon` | PPO clipping coefficient |
| `maximum_gradient_normalization` | Gradient clipping threshold |

The total capacity of the dual replay buffer is `2 × buffer_capacity`: one part stores high-quality trajectories, and the other stores recent trajectories.

## Outputs

For an output folder named `erlplace_adaptec1`, the training process creates:

```text
results/erlplace_adaptec1/
├── log.log                  # Runtime log
├── models/                  # Reserved for model artifacts; current script does not write checkpoints
├── placements/              # Best placement files in UCLA `.pl` format
└── solutions/               # Pickled best placement solutions

tensorboard/erlplace_adaptec1/
└── events.out.tfevents.*    # TensorBoard event files
```

The best placement is saved in both pickle and `.pl` formats. The `.pl` file can be inspected with standard macro-placement visualization tools.

To monitor training with TensorBoard:

```bash
tensorboard --logdir ./tensorboard
```

## Reproducing Paper Experiments

1. Install the required Python and PyTorch dependencies.
2. Confirm that the selected ISPD2005 benchmark exists under `benchmarks/`.
3. Review or modify the corresponding file in `configs/`.
4. Run `main.py` from the repository root with explicit seed, GPU, benchmark, and output folder arguments.
5. Compare the generated placement and training logs with the experimental settings reported in the paper.

A typical experiment command is:

```bash
python main.py \
    -s 42 \
    -g 0 \
    -b adaptec3 \
    -f paper_adaptec3_seed42 \
    -l info
```

For multi-seed experiments, use different values for `--seed` and separate output folders for each run.

## Citation

If you find this code useful, please cite the corresponding paper:

```bibtex
@misc{zhu2026erlplace,
  title   = {An Evolutionary Reinforcement Learning Framework for Macro Placement Optimization},
  author  = {Zhu, Qingling and Hu, Ziyang and Lin, Qiuzhen and Coello Coello, Carlos A. and Mao, Junfa and Li, Jianqiang},
  year    = {2026},
  note    = {Manuscript}
}
```

Please update the BibTeX entry if the paper is published in a journal or conference proceedings.

## Acknowledgements

This implementation builds on the macro placement environment and network design of EfficientPlace. The experiments use the ISPD2005 benchmark suite. We thank the maintainers and contributors of these resources.

## License

This project is released under the GPL-3.0. See [LICENSE](LICENSE) for details.