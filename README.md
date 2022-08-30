# IMLE

* Conditional IMLE refactoring CAM-NET

* Unconditional IMLE with Gaussian Mixture noise


# Conditional IMLE refactoring CAM-NET

Current CAM-NET has triple loops when training model and this makes code complicated. The goal here is to improve usability and complexity of CAM-NET model training procedure by reducing triple loops to double loops while keeping the performance of CAM-NET.

[Original CAM-NET](https://niopeng.github.io/CAM-Net/) | [Paper](https://arxiv.org/abs/2106.09015)

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
python data/SetupDataset.py --data DATASET_NAME --sizes [16, 32, 64, 128, 256]
```
DATASET_NAME options: ["tinyImagenet", "miniImagenet", "camnet3", "cifar10", "camnet3_lmdb", "camnet3_deci_lmdb", "camnet3_centi_lmdb"]

This may take some time, so please download dataset first.

### Run Code

```
python TrainGeneratorWandB.py --data_tr DATASET_NAME --data_val DATASET_NAME --res 16 ... 256
```

### TrainGeneratorWandB Arguments
```
  --wandb {disabled,online,offline}
                        disabled: no W&B logging, online: normal W&B logging
  --suffix SUFFIX       optional training suffix
  --job_id JOB_ID       Variable for storing SLURM job ID
  --uid UID             Unique identifier for the run. Should be specified only when resuming, as it needs to be generated via WandB otherwise
  --resume RESUME       a path or epoch number to resume from or nothing for no resuming
  --data_path DATA_PATH
                        path to where datasets are stored
  --spi SPI             samples per image in logging, showing the model's diversity.
  --num_val_images NUM_VAL_IMAGES
                        Number of images to use for validation
  --chunk_epochs {0,1}  whether to chunk by epoch. Useful for ComputeCanada, annoying otherwise.
  --gpus GPUS [GPUS ...]
                        GPU ids
  --code_bs CODE_BS     GPU ids
  --data_tr DATA_TR     data to train on
  --data_val DATA_VAL   data to train on
  --res RES [RES ...]   resolutions to see data at
  --alpha ALPHA         Amount of weight on MSE loss
  --seed SEED           random seed
  --proj_dim PROJ_DIM   projection dimensionality
  --epochs EPOCHS       number of epochs (months) to train for
  --outer_loops OUTER_LOOPS
                        number of outer_loops to train for
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
  --grayscale {0,0.5,1}
                        grayscale corruption
  --mask_res MASK_RES   sidelength of image at which to do masking
  --mask_frac MASK_FRAC
                        fraction of pixels to mask
  --fill {color,zero}   how to fill masked out areas
  --code_nc CODE_NC     number of code channels
  --in_nc IN_NC         number of input channels. SHOULD ALWAYS BE THREE.
  --out_nc OUT_NC       number of output channels
  --map_nc MAP_NC       number of input channels to mapping net
  --latent_nc LATENT_NC
                        number of channels inside the mapping net
  --resid_nc RESID_NC [RESID_NC ...]
                        list of numbers of residual channels in RRDB blocks for each CAMNet level
  --dense_nc DENSE_NC [DENSE_NC ...]
                        list of numbers of dense channels in RRDB blocks for each CAMNet level
  --n_blocks N_BLOCKS   number of RRDB blocks inside each level
  --act_type {leakyrelu}
                        activation type
  --init_type {kaiming,normal}
                        NN weight initialization method
  --init_scale INIT_SCALE
                        Scale for weight initialization
```

---

## Key Objects

* KorKMinusOne
* CIMLEDataLoader

## KorKMinusOne

KorKMinusOne (KKM), is to track of how many times each data has been used. 

### How KorKMinusOne works

```
class KorKMinusOne
INPUT: idxs(list of index), shuffle

    idxs -> idxs, counter -> 0, shuffle -> shuffle
    
    function pop
        
        if counter reaches to end of idxs 
           then counter -> 0
           
           if shuffle = True then
              randomize the elements position in idxs
        
        result -> idxs[counter]
        counter -> counter + 1

        return result
```

`idxs` input is a list that maps each data's positional index. Example:
```
kkm = KorKMinusOne(range(len(data_tr)), shuffle=True)
```
`shuffle` is to dertmine whether to randomize `idxs` at each `epoch`. `shuffle = False` by default.

## CIMLEDataLoader

CIMLEDataLoader is a iterator object that subsamples data and returns chained dataloaders lazily, and the number of chained dataloaders are determined by `subsample_size` and `num_iteration` arguments.

Another apporach to implement CIMLEDataLoader was to modify DataLoader source code to sample dataset when `iter` method gets called. However, `len` of dataloader is initialized in `__init__` and updating length of dataloader dynamically as chained dataloader generate was making conflict with other inner methods that are utilizing `__len__()` method. Thus, instead, iterator object has been created. 

### How CIMLEDataLoader works

#### Initialization
```
def __init__(self, dataset, kkm, model, corruptor, z_gen, loss_fn, num_samples, sample_parallelism, code_bs, subsample_size=None, num_iteration=1, pin_memory: bool = False, shuffle: Optional[bool] = None, batch_size: Optional[int] = 1, num_workers: int = 0, drop_last: bool = False)
```

CIMLE DataLoader takes additional arguments upon normal Dataloader. kkm, model, corruptor, z_gen, loss_fn, num_samples, sample_parallelism, code_bs, subsample_size, num_iteration are additionals. Key arguments here are `kkm`, `num_iteration`, and `subsample_size`.

During initialization, `num_chained_loaders` determines how many chained dataloader to be generated. This is calculated as follows:

```
if num_iteration is greater than (subsample_size // batch_size)          # subsample_size // batch_size is how many batch data can be fitted             
    then num_chained_loaders -> num_iteration // (subsample_size // batch_size)
else
    num_chained_loaders -> 1
```

By looping through range of `num_chained_loaders`, DataLoaders with `subsample_size // batch_size` iterations are generated and appended into a list. When the loop reached to the last one, it checkes `num_iteration % (subsample_size // batch_size) != 0`. 

This checks whether the `num_iteration` evenly divisible by number of iterations for each subsample `(subsample_size // batch_size)`. If it is divisible, then another DataLoader is generated as previous, otherwise, a DataLoader with `num_iteration % (subsample_size // batch_size)` amounts of iterations is generated which is less iterations than previously generated DataLoaders and appended to the list.

At last, list of DataLoaders will be chained by `itertools.chain` method. 

## Improvment

By using `CIMLEDataLoader`, user will see less loop complexity, have better readability, and usability.

**previous training structure**
```
loader = DataLoader(.....)
for epoch in epochs

    for (x, ys) in loader
        
        #### Sampling ####
        batch_loader = DataLoader(.....)
        
        for (x, latent, ys) in batch_loader
        
            #### Training ####
```

**updated strcuture**
```
k_or_k_minus_one = KorKMinusOne(range(len(data_tr)), shuffle=True)
loader = CIMLEDataLoader(dataset, k_or_k_minus_one,  model, corruptor, z_gen, loss_fn, ...)

for loop in range(args.outer_loop)

    for (x, latent, ys) in loader        #### Sampling lazily ####
        #### Training ####
        
```
