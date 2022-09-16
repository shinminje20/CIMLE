"""Downloads and sets up a dataset."""

import argparse
import zipfile
import gdown
from tqdm import tqdm
from DataUtils import *
import shutil

def gdown_unzip(url, result):
    """Downloads the file at Google drive URL [url], unzips it, and removes any
    hidden files that are not `.` and `..`.
    """
    zip_path = f"{data_dir}/{result}.zip"
    gdown.download(url, zip_path, quiet=False)
    tqdm.write("Unzipping dataset...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(path=data_dir)
    os.remove(zip_path)
    tqdm.write("Removing potentially bad files...")
    remove_bad_files(data_dir)

    return f"{data_dir}/{result}"

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Dataset downloading and creation")
    P.add_argument("--also_cls_first", action="store_true",
        help="also make a class-first dataset split")
    P.add_argument("--sizes", default=[16, 32, 64, 128, 256], type=int,
        nargs="+",
        help="sizes to make the images. -1 or zero for no resizing")
    P.add_argument("--use_existing", type=str,
        help="use existing dataset instead of downloading one")
    args = P.parse_args()

    tqdm.write(f"----- Downloading base dataset -----")
    dataset_dir = gdown_unzip("https://drive.google.com/u/1/uc?id=1Yly8SJitGnA25opej57TRLlSyoGmqeIN&export=download", "camnet3")
    data_dir, _ = os.path.split(dataset_dir)
    new_dataset_dir = data_dir + '/' + "butterfly"
    
    try: 
        shutil.rmtree(new_dataset_dir)
    except FileNotFoundError:
        pass
    os.rename(dataset_dir, new_dataset_dir)

    paths_to_remove = []
    for d1 in os.listdir(new_dataset_dir):
        p1 = os.path.join(new_dataset_dir, d1)
        for i, item in enumerate(os.listdir(p1)):
            if i != 1:
                paths_to_remove.append(os.path.join(p1,item))

    for item in paths_to_remove:
        shutil.rmtree(item)

    
    if len(args.sizes) == 1 and args.sizes[0] <= 0:
        tqdm.write("----- Not resizing dataset -----")
        all_datasets = [new_dataset_dir]
    else:
        tqdm.write(f"----- Generating new resolutions: {args.sizes} -----")
        all_datasets = [new_dataset_dir] + [resize_dataset(new_dataset_dir, s)
            for s in tqdm(args.sizes)]
        
    if args.also_cls_first and not "lmdb" in "camnet3":
        tqdm.write(f"----- Making class-first copies -----")
        for dataset in all_datasets:
            make_cls_first(dataset)
            tqdm.write(f"Made a class-first copy of {dataset}")
