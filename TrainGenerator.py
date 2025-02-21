import argparse
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import wandb

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset

from ConditionalIMLE import CIMLEDataLoader, KorKMinusOne
from CAMNet import *
from Corruptions import Corruption
from Data import *
from Losses import *
from utils.Utils import *
from utils.UtilsColorSpace import *
from utils.UtilsNN import *

from functools import partial

def get_z_dims(args):
    """Returns a list of random noise dimensionalities for one sampling."""
    return [(args.map_nc + args.code_nc * r ** 2,) for r in args.res[:-1]]
#  num_of_input_channels_to_mapping_net * num_of_code_channel * 64^2
mm = None
def get_z_gen(z_dims, bs, level=0, sample_method="normal", input=None, num_components=5,  **kwargs):
    """Returns a latent code for a model.

    Args:
    z_dims          -- list of tuples of shapes to generate
    bs              -- batch size to generate for
    level           -- the level to generate a shape for or all for 'all' to get
                        a list of codes, one for each level
    sample_method   -- the method to use to sample
    input           -- input for test-time sampling
    num_components  -- number of components for mixture-based sampling
    """
    if sample_method == "normal":
        if level == "all":
            return [torch.randn((bs,) + dim) for dim in z_dims]
        else:
            return torch.randn((bs,) + z_dims[level])
    elif sample_method == "mixture":
        global mm
        if mm is None:
            mm = [torch.rand(1, num_components, *dim) for dim in z_dims]
            mm = [nn.functional.normalize(m, dim=2) for m in mm]

        if input is None:
            idxs = torch.tensor(random.choices(range(num_components), k=bs))
        elif input == "show_components":
            idxs = torch.tensor([i % num_components for i in range(bs)])
        elif isinstance(input, torch.Tensor):
            idxs = input
        else:
            pass

        neg_ones = [[-1] * (1 + len(dim)) for dim in z_dims]
        if level == "all":
            means = [mm[level].expand(bs, *neg_ones[level])[torch.arange(bs), idxs] for level in range(len(mm))]
            return [m + torch.randn(m.shape) for m in means]
        else:
            means = mm[level].expand(bs, *neg_ones[level])[torch.arange(bs), idxs]
            return means + torch.randn(means.shape)
    else:
        raise NotImplementedError()

def validate(corruptor, model, z_gen, loader_eval, loss_fn, args):
    """Returns a list of lists, where each sublist contains first a ground-truth
    image and then [samples_per_image] images conditioned on that one.

    Args:
    corruptor   -- a corruptor to remove information from images
    model       -- a model to fix corrupted images
    z_gen       -- noise generator for [model]
    loader_eval -- dataloader over evaluation data
    loss_fn     -- loss function for one CAMNet level
    args        -- argparse arguments for the run

    Returns:
    results     -- 2D grid of images to show
    loss        -- average (LPIPS, MSE, Resolution) losses for the images.
                    Because it's computed over only the last level, the
                    Resolution loss will be less than recorded training loss
    """
    results = []
    lpips_loss, mse_loss, combined_loss = 0, 0, 0
    with torch.no_grad():
        for x,y in tqdm(loader_eval, desc="Generating samples", leave=False, dynamic_ncols=True):
            bs = len(x)
            cx = corruptor(x)
            cx_expanded = cx.repeat_interleave(args.spi, dim=0)
            codes = z_gen(bs * args.spi, level="all", input="show_components")
            outputs = model(cx_expanded, codes, loi=-1)
            lpips_loss_, mse_loss_, combined_loss_ = loss_fn(outputs, y[-1], return_metrics=True)
            outputs = outputs.view(bs, args.spi, 3, args.res[-1], args.res[-1])

            idxs = torch.argsort(combined_loss_.view(bs, args.spi), dim=-1)
            outputs = outputs[torch.arange(bs).unsqueeze(1), idxs]            
            images = [[s for s in samples] for samples in outputs]
            images = [[y_, c] + s for y_,c,s in zip(y[-1], cx, images)]

            lpips_loss += lpips_loss_.mean().item()
            mse_loss += mse_loss_.mean().item()
            combined_loss += combined_loss_.mean().item()
            results += images

    return results, lpips_loss / len(loader_eval), mse_loss / len(loader_eval), combined_loss / len(loader_eval)

def get_args(args=None):
    P = argparse.ArgumentParser(description="CAMNet training")
    # Non-hyperparameter arguments. These aren't logged!
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="disabled: no W&B logging, online: normal W&B logging")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--job_id", default=None, type=str,
        help="Variable for storing SLURM job ID")
    P.add_argument("--uid", default=None, type=str,
        help="Unique identifier for the run. Should be specified only when resuming, as it needs to be generated via WandB otherwise")
    P.add_argument("--resume", type=str, default=None,
        help="a path or loop number to resume from or nothing for no resuming")

    P.add_argument("--data_path", default=data_dir, type=str,
        help="path to where datasets are stored")
    P.add_argument("--spi", type=int, default=6,
        help="samples per image in logging, showing the model's diversity.")
    P.add_argument("--num_val_images", type=int, default=10,
        help="Number of images to use for validation")
    P.add_argument("--chunk_loops", type=int, choices=[0, 1], default=0,
        help="whether to chunk by loop. Useful for ComputeCanada, annoying otherwise.")
    P.add_argument("--gpus", type=int, default=[0, 1], nargs="+",
        help="GPU ids")
    P.add_argument("--code_bs", type=int, default=2,
        help="GPU ids")

    # Corruption hyperparameter arguments
    P.add_argument("--grayscale", default=0, type=float, choices=[0, 1],
        help="grayscale corruption")

    # Training hyperparameter arguments. These are logged!
    P.add_argument("--data_tr", type=is_valid_data, required=True,
        help="data to train on")
    P.add_argument("--data_val", type=is_valid_data, required=True,
        help="data to train on")
    P.add_argument("--res", nargs="+", type=int, default=[64, 64, 64, 64, 128],
        help="resolutions to see data at")
    P.add_argument("--alpha", type=float, default=.1,
        help="Amount of weight on MSE loss")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--proj_dim", default=1000, type=int,
        help="projection dimensionality")
    P.add_argument("--outer_loops", default=40, type=int,
        help="outer_loop argument is used instead of `epoch`. Each `outer_loop` iteration is one subsampling.")
    P.add_argument("--bs", type=int, default=512,
        help="batch size")
    P.add_argument("--mini_bs", type=int, default=8,
        help="batch size")
    P.add_argument("--ns", type=int, nargs="+", default=[128],
        help="number of samples for IMLE")
    P.add_argument("--ipc", type=int, default=10240,
        help="Effective gradient steps per set of codes. --ipc // --mini_bs is equivalent to num_days in the original CAMNet formulation")
    P.add_argument("--lr", type=float, default=1e-4,
        help="learning rate")
    P.add_argument("--color_space", choices=["rgb", "lab"], default="rgb",
        help="Color space to use during training")
    P.add_argument("--sp", type=int, default=[128], nargs="+",
        help="parallelism across samples during code training")
    P.add_argument("--subsample_size", default=None, type=int,
        help="number of subsample data ")
    P.add_argument("--num_iteration", default=1, type=int,
        help="number of subsample data ")
        
    P.add_argument("--sample_method", choices=["normal", "mixture"], default="normal",
        help="The method with which to sample latent codes")

    # Model hyperparameter arguments
    P.add_argument("--code_nc", default=5, type=int,
        help="number of code channels")
    P.add_argument("--map_nc", default=128, type=int,
        help="number of input channels to mapping net")

    args = P.parse_args() if args is None else P.parse_args(args)
    args.levels = len(args.res) - 1
    args.ns = make_list(args.ns, length=args.levels)
    args.sp = make_list(args.sp, length=args.levels)

    # Make sure we won't break sampling.
    assert args.bs % len(args.gpus) == 0
    for ns,sp in zip(args.ns, args.sp):
        if not (ns * sp) % len(args.gpus) == 0:
            raise ValueError(f"number of samples * sample parallelism must be a multiple of the number of GPUS for each level")
    args.spi = args.spi - (args.spi % len(args.gpus))

    assert args.code_bs >= len(args.gpus)

    if not args.ipc % args.mini_bs == 0 or args.ipc // args.mini_bs == 0:
        raise ValueError(f"--ipc should be a multiple of --mini_bs")

    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    return args

if __name__ == "__main__":
    args = get_args()

    ############################################################################
    # Handle resuming.
    ############################################################################
    save_dir = generator_folder(args)
    print("save directory : ", save_dir)
    if str(args.resume).isdigit():
        args.resume = int(args.resume) - 1
        if int(args.resume) == -1:
            resume_file = None
        elif os.path.exists(f"{save_dir}/{args.resume}.pt"):
            resume_file = f"{save_dir}/{args.resume}.pt"
        else:
            raise ValueError(f"File {save_dir}/{args.resume}.pt doesn't exist")
    elif isinstance(args.resume, str):
        resume_file = args.resume
    else:
        resume_file = None

    if resume_file is None:
        save_dir = generator_folder(args)
        cur_seed = set_seed(args.seed)

        # Setup the experiment. Importantly, we copy the experiment's ID to
        # [args] so that we can resume it later.
        args.run_id = wandb.util.generate_id()
        wandb.init(anonymous="allow", id=args.uid, config=args,
            mode=args.wandb, project="isicle-generator",
            name=save_dir.replace(f"{project_dir}/generators/", ""))
        corruptor = Corruption(**vars(args))
        model = nn.DataParallel(CAMNet(**vars(args)), device_ids=args.gpus).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        last_loop = -1
    else:
        tqdm.write(f"Resuming from {resume_file}")
        resume_data = torch.load(resume_file)

        # Copy non-hyperparameter information from the current arguments to the
        # ones we're resuming
        curr_args = args
        args = resume_data["args"]
        args.data_path = curr_args.data_path
        args.gpus = curr_args.gpus
        args.chunk_loops = curr_args.chunk_loops
        args.wandb = curr_args.wandb
        save_dir = generator_folder(args)
        cur_seed = set_seed(resume_data["seed"])

        wandb.init(id=args.uid, resume="must", mode=args.wandb,
            project="isicle-generator", config=args,
            name=save_dir.replace(f"{project_dir}/generators/", ""))

        model = resume_data["model"].to(device)
        optimizer = resume_data["optimizer"]
        corruptor = resume_data["corruptor"].to(device)
        last_loop = resume_data["last_loop"]
        scheduler = resume_data["scheduler"]
        k_or_k_minus_one = resume_data["k_or_k_minus_one"]

    # Set up the loss function
    loss_fn = nn.DataParallel(ResolutionLoss(alpha=args.alpha), device_ids=args.gpus).to(device)
    
    data_tr, data_val = get_imagefolder_data(args.data_tr, args.data_val,
        res=args.res,
        data_path=args.data_path)

    data_tr = GeneratorDataset(data_tr, get_gen_augs(args))

    if args.data_val == "cv":
        step = int((len(data_tr) / args.num_val_images) + .5)
        idxs_val = {idx for idx in range(0, len(data_tr), step)}

        if len(idxs_val) == len(data_tr):
            raise ValueError(f"Too many validation images selected; no data is left for training. Reduce --num_val_images to below {len(data_tr) // 2}")

        idxs_tr = [idx for idx in range(len(data_tr)) if not idx in idxs_val]
        data_tr = Subset(data_tr, indices=idxs_tr)
        data_val = Subset(data_tr, indices=list(idxs_val))
    else:
        data_val = GeneratorDataset(data_val, get_gen_augs(args))
        step = int((len(data_val) / args.num_val_images) + .5)
        idxs_val = {idx for idx in range(0, len(data_val), step)}
        data_val = Subset(data_val, indices=list(idxs_val))
    

    z_gen = partial(get_z_gen, get_z_dims(args),
        sample_method=args.sample_method)   
    
    loader_eval = DataLoader(data_val,
        shuffle=False,
        batch_size=max(len(args.gpus), args.bs),
        num_workers=8,
        drop_last=True)

    ########################################################################
    # Construct the scheduler—strictly speaking, constructing it makes no sense
    # here, but we need to do it only if we're starting a new run.
    ########################################################################

    tqdm.write(f"----- Final Arguments -----")
    tqdm.write(dict_to_nice_str(vars(args)))
    tqdm.write(f"----- Beginning Training -----")
    
    if resume_file is None:
        k_or_k_minus_one = KorKMinusOne(range(len(data_tr)), shuffle=True)
        scheduler = CosineAnnealingLR(optimizer,
            args.outer_loops * args.num_iteration,
            eta_min=1e-4,
            last_epoch=max(-1, last_loop * args.num_iteration))


    loader_tr = CIMLEDataLoader(data_tr, k_or_k_minus_one,  model, corruptor, z_gen, loss_fn, args.ns, args.sp, args.code_bs,
                    subsample_size=args.subsample_size,
                    num_iteration=args.num_iteration,
                    pin_memory=True,
                    shuffle=True,
                    batch_size=max(len(args.gpus), args.bs),
                    num_workers=8,
                    drop_last=True)

    cur_step = 0

    end_loop = last_loop + 2 if args.chunk_loops else args.outer_loops
    cur_step = (last_loop + 1) * args.num_iteration
    tqdm.write(f"LOG: Running loops indexed {last_loop + 1} to {end_loop}")

    for loop in tqdm(range(last_loop + 1, end_loop),
        desc="OuterLoops",
        dynamic_ncols=True):
        
        for batch_idx, (cx, codes, ys) in tqdm(enumerate(loader_tr),
            desc="Batches",
            leave=False,
            dynamic_ncols=True,
            total=len(loader_tr)):

            batch_loss = 0
            fx = model(cx.to(device), [c.to(device) for c in codes])
            loss = compute_loss_over_list(fx, [y.to(device) for y in ys], loss_fn)

            del codes, cx
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            batch_loss += loss.detach()
            cur_step += 1
            wandb.log({
                "batch loss": loss.detach(),
                "learning rate": get_lr(scheduler)[0]
            }, step=cur_step)
            
            del ys, loss, batch_loss

            
        ####################################################################
        # Log data after each loop
        ####################################################################
        images_val, lpips_loss_val, mse_loss_val, comb_loss_val = validate(
            corruptor, model, z_gen, loader_eval, loss_fn, args)
        images_file = f"{save_dir}/val_images/step{(loop + 1) * len(loader_tr)}.png"
        save_image_grid(images_val, images_file)
        wandb.log({
            "Loop_LPIPS loss": lpips_loss_val,
            "Loop_MSE loss": mse_loss_val,
            "Loop_combined loss": comb_loss_val,
            "Loop_generated images": wandb.Image(images_file),
        }, step=cur_step)

        tqdm.write(f"Loop {loop:3}/{args.outer_loops} | Loop_lr {get_lr(scheduler)[0]:.5e} | Loop_loss_val {comb_loss_val:.5e}")

        del images_val, lpips_loss_val, mse_loss_val, comb_loss_val

        save_checkpoint({"corruptor": corruptor.cpu(), "model": model.cpu(),
            "last_loop": loop, "args": args, "scheduler": scheduler,
            "optimizer": optimizer, "k_or_k_minus_one": k_or_k_minus_one}, f"{save_dir}/{loop}.pt")
        corruptor, model = corruptor.to(device), model.to(device)