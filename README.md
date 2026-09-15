# Flow-based Domain Randomization

[Flow-based Domain Randomization for Learning and Sequencing Robotic Skills](https://arxiv.org/pdf/2502.01800)


## How to run

Set up and activate a conda environment that works with [isaaclab](https://isaac-sim.github.io/IsaacLab/main/index.html)

```
python -m pip install -e .
```

To run, simply execute the following command from the root directory

```
python train_rl.py --task=<env>-<method>-v0 --headless
```

where env in `{Ant, Anymal, Cartpole, Humanoid, Gears, Quadcopter}` and method in `{GOFLOW, DORAEMON, LSDR, FullDR, ADR, NoDR}`

For example, 

```
python train_rl.py --task=Gears-GOFLOW-v0 --headless
```

Remove `--headless` to visualize. For the gears task, you will need to reduce the number of environments to visualize.
To reproduce the experiments in the paper, see `goflow/main_experiments.py`


All checkpoints and tensorboard logs get saved to the `logs` directory. You can visualize a training checkpoint with

```
python train_rl.py --task=<env>-<method>-v0 --checkpoint=<path-to-checkpoint>
```

## Cite this paper
```
@inproceedings{curtis2025flowbaseddomainrandomizationlearning,
  title     = {Flow-based Domain Randomization for Learning and Sequencing Robotic Skills},
  author    = {Aidan Curtis and Eric Li and Michael Noseworthy and Nishad Gothoskar and Sachin Chitta and Hui Li and Leslie Pack Kaelbling and Nicole Carey},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {To appear},
  url       = {https://arxiv.org/abs/2502.01800},
  archivePrefix = {arXiv},
  eprint    = {2502.01800},
  primaryClass = {cs.RO}
}
```


## Local reproduction workspace

This clean clone preserves upstream commit `a8c6af5de7f427418783fd9faa20d50f38b734a9`. Exact environment and execution commands are in [the reproduction README](experiments/connector_handoff/README.md). The original-task pilots ran, but the baseline competence gate remains unmet; the connector experiment has not been completed. See [project status](docs/project_status.md) and [method fidelity](docs/method_fidelity.md).
