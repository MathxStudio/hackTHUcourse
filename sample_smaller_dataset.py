import os
import random
import shutil
import math

from tqdm import tqdm
from consts import *

def sample_files(src_dir, dest_dir, percent):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    file_count = sum(len(files) for _, _, files in os.walk(src_dir))
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(src_dir):
            relative_path = os.path.relpath(root, src_dir)
            dest_subdir = os.path.join(dest_dir, relative_path)

            os.makedirs(dest_subdir, exist_ok=True)

            if files:
                num_files_to_sample = math.ceil(len(files) * (percent / 100))

                sampled_files = random.sample(files, num_files_to_sample)

                for file in sampled_files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_subdir, file)
                    shutil.copy2(src_file, dest_file)
                pbar.update(len(files))

def sample_train_test(path_to_full_data, percent):
    print("Sampling data into training & testing set...")
    all_path = os.path.join(path_to_full_data, 'all')
    train_path = os.path.join(path_to_full_data, 'train')
    test_path = os.path.join(path_to_full_data, 'test')

    if not os.path.exists(all_path):
        raise FileNotFoundError(f"The 'all' directory does not exist in {path_to_full_data}")

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    os.makedirs(train_path)
    os.makedirs(test_path)
    subdirs = [d for d in os.listdir(all_path) if os.path.isdir(os.path.join(all_path, d))]

    for subdir in tqdm(subdirs, desc="Processing Directories", unit="dir"):
        subdir_path = os.path.join(all_path, subdir)
        if os.path.isdir(subdir_path):
            train_subdir = os.path.join(train_path, subdir)
            test_subdir = os.path.join(test_path, subdir)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)

            all_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            total_files = len(all_files)
            train_count = int(total_files * percent / 100)

            random.shuffle(all_files)

            train_files = all_files[:train_count]
            test_files = all_files[train_count:]

            for file in train_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(train_subdir, file))
            for file in test_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(test_subdir, file))

src_directory = os.path.join(SEGMENTED_presampling, 'all')
dest_directory = os.path.join(SEGMENTED_aftersampling, 'all')
sampling_percentage = 30
train_percentage = 80

if not os.path.exists(dest_directory):
    sample_files(src_directory, dest_directory, sampling_percentage)
if not os.path.exists(os.path.join(dest_directory, 'train')):
    sample_train_test(SEGMENTED_aftersampling, train_percentage)
