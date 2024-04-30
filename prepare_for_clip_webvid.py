import urllib.request
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import tarfile
import logging
import concurrent.futures
import numpy as np

def reporthook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading... {percent}%", end="")
    sys.stdout.flush()

def download_file(url, filename):
    # command = f"wget -t 0 -O {filename} {url}"
    # os.system(command)
    retry_count = 0
    print(f"Downloading {filename} from {url}")
    sys.stdout.flush()
    while retry_count < 3:
        try:
            urllib.request.urlretrieve(url, filename, reporthook)
            break
        except:
            print(f"\nError downloading {filename}. Retrying in 5 seconds...")
            sys.stdout.flush()
            time.sleep(5)
            retry_count += 1
    if retry_count == 3:
        print(f"\nFailed to download {filename} after 3 attempts.")
    else:
        print(f"\nDownloaded {filename} successfully.")

with ThreadPoolExecutor(max_workers=16) as executor:
    for i in range(1088):
        # https://huggingface.co/datasets/iejMac/CLIP-WebVid/resolve/main/data/train/000000000.tar?download=true
        url = f"https://huggingface.co/datasets/iejMac/CLIP-WebVid/resolve/main/data/train/{i:09d}.tar"
        filename = f"./data/clip-webvid-2.5M/temp_tar_data/{i:09d}.tar"
        executor.submit(download_file, url, filename)
        

if not os.path.exists('./data/clip-webvid-2.5M/extracted_all_data'):
    os.makedirs('./data/clip-webvid-2.5M/extracted_all_data')

# logging.basicConfig(filename='extract_all_files.log', level=logging.INFO,
#                     format='%(asctime)s %(levelname)s: %(message)s')

def extract_archive(filename):
    with tarfile.open(filename, 'r') as tar:
        logging.info(f'Extracting files from {filename}...')
        tar.extractall('./data/clip-webvid-2.5M/extracted_all_data', numeric_owner=True)
        logging.info(f'Successfully extracted files from {filename}.')

# loop over all .tar files in the directory
filenames = [filename for filename in os.listdir('./data/clip-webvid-2.5M/temp_tar_data') if filename.endswith('.tar')]
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(extract_archive, filename) for filename in filenames]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logging.error(f'Error extracting files: {e}')
            
# define the path to the all_data directory
all_data_dir = "./data/clip-webvid-2.5M/extracted_all_data"
dim = 512
vectors = []

total_cnt = 0
# loop over all subdirectories in the all_data directory
for subdir in os.listdir(all_data_dir):
    subdir_path = os.path.join(all_data_dir, subdir)
    
    # skip non-directory files
    if not os.path.isdir(subdir_path):
        continue
    
    # loop over all .npy files in the subdirectory
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        
        # skip non-.npy files
        if not file.endswith(".npy"):
            continue
        # print("file_path: ", file_path)
        # load the matrix from the .npy file
        matrix = np.load(file_path).astype(np.float32)
        matrix = matrix.astype(np.float32)

        # normalize each row vector. Then get the mean, refer to paper "clip4clip".
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm[matrix_norm == 0] = 1  # avoid division by zero
        matrix_normed = matrix / matrix_norm
        vector = np.mean(matrix_normed, axis=0)
        vector_norm = np.linalg.norm(vector)
        vector_normed = vector / vector_norm
        
        # add the vector to the list of aggregated vectors
        vectors.append(vector_normed)
        total_cnt = total_cnt + 1
        if total_cnt % 10000 == 0:
            print("total_cnt: ", total_cnt)

        if dim != vector_normed.shape[0]:
            print("dim != vector_normed.shape[0]")
            exit(1)
    # if total_cnt == 1000000:
    #     break
    # 2.5M
    if total_cnt >= 2500000:
        break

# convert the list of vectors to a 2D NumPy array
vectors_array = np.array(vectors, dtype=np.float32)
if dim != vectors_array.shape[1]:
    print("dim != vectors_array.shape[1]")
    exit(1)
#write to file , with meta data: num_vectors, dimension in np.uint32 type, following is the vectors
f_path = "./data/clip-webvid-2.5M/base.2.5M.fbin"
f = open(f_path, "wb")
# little-end
f.write(total_cnt.to_bytes(4, byteorder='little', signed=False))
# dim
f.write(dim.to_bytes(4, byteorder='little', signed=False))

vectors_array.tofile(f)
f.close()

# print("total_cnt: ", total_cnt)
# print("dim: ", dim)
# print("vectors_array.shape: ", vectors_array.shape)
# print("vectors_array.dtype: ", vectors_array.dtype)
# #file name:
# print("file name: ", f_path)

# download text embeddings, or you can encode the texts by CLIP model locally (needed to be normalized).
url = "https://zenodo.org/api/records/11073098/draft/files/webvid.query.train.2.5M.fbin/content"
filename = "./data/clip-webvid-2.5M/query.train.2.5M.fbin"
download_file(url, filename)