import os
import shutil
import tqdm 

def clone_directory_with_pickles(source_dir, destination_dir):
    # Walk through the source directory
    for root, dirs, files in tqdm.tqdm(os.walk(source_dir)):
        # Calculate the relative path
        relative_path = os.path.relpath(root, source_dir)
        # Create the corresponding directory in the destination
        new_dir = os.path.join(destination_dir, relative_path)
        os.makedirs(new_dir, exist_ok=True)
        
        # Copy only pickle files
        for file in tqdm.tqdm(files):
            if file.endswith('.pkl'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(new_dir, file)
                shutil.copy2(source_file, destination_file)

# Example usage
source_directory = '/scratch/tathagato/naacl/'
destination_directory = '/scratch/tathagato/naacl_pickle_files'

clone_directory_with_pickles(source_directory, destination_directory)