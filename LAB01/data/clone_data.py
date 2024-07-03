import os
import shutil
from git import Repo

def clone_repo(repo_url, clone_dir):
    if os.path.exists(clone_dir):
        print(f"Directory {clone_dir} already exists. Removing it.")
        shutil.rmtree(clone_dir, onerror=remove_readonly)
    try:
        Repo.clone_from(repo_url, clone_dir)
        print(f"Successfully cloned repository {repo_url}")
    except Exception as e:
        print(f"Error cloning repository {repo_url}: {e}")

def copy_data(src_dir, dest_dir, repo_name):
    # Ensure the destination directory is unique
    dest_path = os.path.join(dest_dir, repo_name)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{repo_name}_{counter}")
        counter += 1

    if os.path.exists(src_dir):
        shutil.copytree(src_dir, dest_path)
        print(f"Successfully copied data from {src_dir} to {dest_path}")
    else:
        print(f"Directory {src_dir} does not exist")

def read_repo_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def remove_readonly(func, path, exc_info):
    os.chmod(path, 0o777)
    func(path)

def main(repo_list_file, base_clone_dir, dest_data_dir):
    repo_list = read_repo_list(repo_list_file)
    
    if not os.path.exists(base_clone_dir):
        os.makedirs(base_clone_dir)
    if not os.path.exists(dest_data_dir):
        os.makedirs(dest_data_dir)
    
    for repo_url in repo_list:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        clone_dir = os.path.join(base_clone_dir, repo_name)
        
        clone_repo(repo_url, clone_dir)
        
        src_data_dir = os.path.join(clone_dir, 'LAB01', 'data')
        if os.path.exists(src_data_dir):
            copy_data(src_data_dir, dest_data_dir, repo_name)
        else:
            print(f"Directory {src_data_dir} does not exist in repository {repo_url}")
        
        # Clean up cloned repository after copying data
        try:
            shutil.rmtree(clone_dir, onerror=remove_readonly)
            print(f"Successfully removed directory {clone_dir}")
        except Exception as e:
            print(f"Error removing directory {clone_dir}: {e}")

if __name__ == "__main__":
    repo_list_file = 'linkrepo.txt'  # Path to the file containing the list of repository links
    base_clone_dir = 'cloned_repos'  # Temporary directory to clone repositories
    dest_data_dir = 'collected_data'  # Destination directory to store data

    main(repo_list_file, base_clone_dir, dest_data_dir)
