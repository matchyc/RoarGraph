# ./data/t2i-10M/query.train.10M.fbin overwrite the first 4 bytes by 10000000 as uint32

import os
import sys
import numpy

def fit_meta_data_in_file(data_file, data_size):
    with open(data_file, 'r+b') as f:
        f.seek(0)
        f.write(numpy.array([data_size], dtype=numpy.uint32).tobytes())
        
if __name__ == '__main__':
    data_file = sys.argv[1]
    data_size = int(sys.argv[2])
    fit_meta_data_in_file(data_file, data_size)
    print(f"change meta data in file {data_file} to {data_size}")
