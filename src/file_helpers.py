import os
import json
from datetime import datetime


def check_subfolders(directory):
    for root, dirs, files in os.walk(directory):
        # Check only subfolders, not the root directory
        if root != directory:
            # Get the number of files in the current folder
            num_files = len(files)
            
            # Check if the number of files is not equal to 10
            if num_files != 10:
                print(f"Folder '{os.path.relpath(root, directory)}' has {num_files} files instead of 10.")


def gather_files(parent_folder:str): 
    """
    Recursively gather all files in the parent folder and its subfolders.

    Parameters:
    - parent_folder (str): The path to the parent folder.

    Returns:
    - List[str]: A list of file paths.
    """
    files_list = []

    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)

    return files_list

    
def custom_file_sort(file_paths,file_type='minhash',sort_criteria='shard'):
    #/scratch/project_462000353/data/redpajama-v2/minhash-2023-14/4832/en_middle.minhash.parquet
    def sort_shard_text(item):
        return int(item[59:63])
    def sort_lang_text(item):
        return item[64:66]
    def sort_lang_min(item):
        return item[66:68]
    
    def sort_lang_dup(item):
        return item[69:71]
            
    if sort_criteria=='shard' and file_type == 'text':
        sorted_items = sorted(file_paths, key=sort_shard_text)
    elif sort_criteria=='lang' and file_type == 'text':
        sorted_items = sorted(file_paths,key=sort_lang_text)
    elif sort_criteria=='lang' and file_type == 'minhash':
        sorted_items = sorted(file_paths, key=sort_lang_min)
    elif sort_criteria=='lang' and file_type =='duplicates':
        sorted_items = sorted(file_paths, key=sort_lang_dup)
    else:
        raise NotImplementedError("File type or sort criteria are not matching!")
    
    return sorted_items


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def format_duration(seconds):
    """Format seconds into nice output

    Args:
        seconds (int): seconds

    Returns:
        str: formatted string
    """    
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

if __name__ == "__main__":
    text_files = "/scratch/project_462000353/data/redpajama-v2/texts-2023-14"
    files = gather_files(text_files)
    sorted_text_files = custom_file_sort(files,file_type='text',sort_criteria='lang')
    print("List of files:")
    for i,file in enumerate(sorted_text_files):
        print(file)
        if i == 5:
            break
    