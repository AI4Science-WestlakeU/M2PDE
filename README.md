# project_module
This is a project module for a new project, which inludes the file framework, DL pipline, RM.md example and some useful utils.
**To be updating now.**

# paper_name

[Paper](URL) | [arXiv](URL) | [Poster](URL) | [Tweet](URL)

Official repo for the paper [Paper Name](URL).<br />
[Author name]()
ICLR 2025 **under review**.

we propose a compositional generative model for multiphysics and multi-component simulation based on diffusion model (MultiSimDiff). MultiSimDiff utilizes models trained on decoupled data for predicting coupled solutions and model trained on small structures for predicting large structures.

Framework of paper:
<a href="url"><img src="./schematic.png" align="center" width="600" ></a>

## Installation


1. Install dependencies.

```code
conda create -n ENV_NAME python=3.x.x
```

Install dependencies:
```code
pip install -r requirements.txt
```

#  file structure
- project_module
  - moose                   # Use to generate datasets for Experiment 2 and Experiment 3, how to use it can be found in: https://mooseframework.org.
  - data                    # data class and dataloader used in the project
  - dataset                 # datasets ready for training or analysis
  - src
    - train                 # codes for training models
    - inference             # codes for inference
    - model                 # model definitions
    - utils                 # Utility scripts and helper functions
    - filepath.py             # Python script for file path handling
  - results                 # results and logs from training and inference
  - .gitignore              # Specifies intentionally untracked files to ignore by git
  - README.md               # Markdown file with information about the project for users
  - reproducibility_statement.md # Markdown file with statements on reproducibility practices
  - requirements.txt        # Text file with a list of dependencies to install


## Dataset and checkpoint

All the dataset can be downloaded in this [this link](https://drive.google.com/file/d/1W30JZzzwsLFyIkWfHKRJeYA_e5JG91zD/view?usp=drive_link). Checkpoints are in [this link](https://drive.google.com/file/d/1tg8eA3v9cx9emutWuDMB_1yax_QmhgLJ/view). Both dataset.zip and checkpoint_path.zip should be decompressed to the root directory of this project.


## Training

Below we provide example commands for training the diffusion model/forward model.
More can be found in "./scripts"

### training model


```code
python reaction_diffusion.py --train_which u --dim 24 --batchsize 256 --paradigm diffusion --epoches 200000
python nuclear_thermal_coupling.py --train_which neutron --dim 8 --batchsize 32 --paradigm diffusion --dataset iter1 --n_dataset 5000 --gradient_accumulate_every 2 --epoches 200000
python heatpipe.py --batchsize 256 --model_type transformer --paradigm surrogate --n_layer 5 --hidden_dim 64 --epoches 100000
```


## inference

The codes for inference are in "./src/inference/"


## Related Projects

* [NAME](URL) (): brief description of the project.

Numerous practices and standards were adopted from [NAME](URL).
## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
    ...
}
```
