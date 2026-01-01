# Imitation: A repository for prototyping imitation learning
This repository provides developers access to simple and modular tools to quickly prototype different architectures and algorithms for imitation learning. We provide a base implementation of several algorithms that can be built upon.

<!-- <b>Motivation.</b> If you’ve ever used robomimic then you might agree with me that it’s too big for its own good. While providing a lot of functionality, it’s hard to quickly spin up new models or training pipelines. This is my attempt to create a unified codebase of popular imitation learning algorithms (eg. diffusion policy, flow policy and more) in a lightweight repository for quick protyping and experimentation. -->

## Installation

```
conda create -n imitation python=3.10
git clone https://github.com/ShivinDass/imitation.git
cd imitation
pip install -r requirements.txt
pip install -e .
```

## Getting Started
This repository follows the hdf5 data format from robomimic. To get started, you can use existing datasets from [robomimic](https://robomimic.github.io/docs/datasets/overview.html) and [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html) directly or convert your own datasets to the correct format.

Choose model config folder at ```config/model/``` and change the data path and output directory. Then run the following script with the experiment name and config path,
```
python scripts/train.py --exp_name debug --config config/model/flow_policy_config.py
```

## Code Structure
```
imitation/
├── algo/            # Algorithm implementations (Recurrent-BC, Flow Policy etc.)
├── models/          # Neural network architectures and building blocks
├── config/          # Configuration files for models and training
├── data/            # Dataset loading and processing
├── evaluators/      # Evaluation frameworks for different environments
├── scripts/         # Training and utility scripts
├── utils/           # Utility functions and helpers
└── wrappers/        # Environment wrappers
```


## TODOs
- Add Data parallel for multi-gpu training
- Add Language modality

## Acknowledgements
This repo takes a lot of insipiration from popular codebases, integrating some of their best features.
1. [Robomimic](https://github.com/ARISE-Initiative/robomimic)
2. [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
3. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

## Citation
```
@software{dass2024imitation,
  author = {Dass, Shivin},
  title = {Imitation: A repository for prototyping imitation learning},
  url = {https://github.com/ShivinDass/imitation/},
  version = {0.1.0},
  year = {2024}
}
```
