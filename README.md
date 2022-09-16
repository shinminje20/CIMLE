
CAM-NET is a unified architecture that can be applied to a broad range of tasks which built on the recently proposed technique of Implicit Maximum Likelihood Estimation (IMLE).

[Original CAM-NET](https://niopeng.github.io/CAM-Net/) | [Paper](https://arxiv.org/abs/2106.09015)


# Differences from the original implementation.

## Improvment

Current CAM-NET has triple loops when training model and this makes code complicated. The goal here is to improve usability and complexity of CAM-NET model training procedure by reducing triple loops to double loops while keeping the performance of CAM-NET.

By using `CIMLEDataLoader`, user will see less loop complexity, have better readability, and usability.

### previous training structure
```
loader = DataLoader(.....)
for epoch in epochs

    for (x, ys) in loader
        
        #### Sampling ####
        batch_loader = DataLoader(.....)
        
        for (x, latent, ys) in batch_loader
        
            #### Training ####
```

### updated strcuture
```
k_or_k_minus_one = KorKMinusOne(range(len(data_tr)), shuffle=True)
loader = CIMLEDataLoader(dataset, k_or_k_minus_one,  model, corruptor, z_gen, loss_fn, ...)

for loop in range(args.outer_loop)

    for (x, latent, ys) in loader        #### Sampling lazily ####
        #### Training ####
        
```

## Quick Start Guide
### Dependencies and Installation
```
Python 3.9
pytorch 1.12.0 
wandb 0.12.17
gdown 4.4.0
tqdm 4.64.0
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```
To clone the repo
```
git clone https://github.com/shinminje20/IMLE.git
```

Prepare and set up the datasets to use. Run
```
python data/SetupDataset.py --sizes 16 32 64 128 256
```

This may take some time, so please download dataset first.

### Run Code

```
python TrainGenerator.py --data_tr butterfly/train --data_val butterfly/val --res 32 64 128 256 --bs 2 --sp 64 32 16 --wandb disabled --code_bs 8 --subsample_size 400 --outer_loops 40 --num_iteration 10000
```

## TrainGenerator Arguments
```
  --wandb {disabled,online,offline}
                        disabled: no W&B logging, online: normal W&B logging
  --suffix SUFFIX       optional training suffix
  --job_id JOB_ID       Variable for storing SLURM job ID
  --uid UID             Unique identifier for the run. Should be specified only when resuming, as it needs to be generated via WandB otherwise
  --resume RESUME       a path or loop number to resume from or nothing for no resuming
  --data_path DATA_PATH
                        path to where datasets are stored
  --spi SPI             samples per image in logging, showing the model's diversity.
  --num_val_images NUM_VAL_IMAGES
                        Number of images to use for validation
  --chunk_loops {0,1}   whether to chunk by loop. Useful for ComputeCanada, annoying otherwise.
  --gpus GPUS [GPUS ...]
                        GPU ids
  --code_bs CODE_BS     GPU ids
  --data_tr DATA_TR     data to train on
  --data_val DATA_VAL   data to train on
  --res RES [RES ...]   resolutions to see data at
  --alpha ALPHA         Amount of weight on MSE loss
  --seed SEED           random seed
  --proj_dim PROJ_DIM   projection dimensionality
  --outer_loops OUTER_LOOPS
                        outer_loop argument is used instead of `epoch`. Each `outer_loop` iteration is one subsampling.
  --bs BS               batch size
  --mini_bs MINI_BS     batch size
  --ns NS [NS ...]      number of samples for IMLE
  --ipc IPC             Effective gradient steps per set of codes. --ipc // --mini_bs is equivalent to num_days in the original CAMNet formulation
  --lr LR               learning rate
  --color_space {rgb,lab}
                        Color space to use during training
  --sp SP [SP ...]      parallelism across samples during code training
  --subsample_size SUBSAMPLE_SIZE
                        number of subsample data
  --num_iteration NUM_ITERATION
                        number of subsample data
  --sample_method {normal,mixture}
                        The method with which to sample latent codes
  --code_nc CODE_NC     number of code channels
  --map_nc MAP_NC       number of input channels to mapping net
```
