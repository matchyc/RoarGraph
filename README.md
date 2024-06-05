# RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search

This repository includes the codes for the VLDB 2024 paper RoarGraph.

![](https://api.visitorbadge.io/api/VisitorHit?user=matchyc&repo=RoarGraph&countColor=%237B1E7A)

[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/matchyc/c4295bccf42f4b2be4b7777a43bd65e9/raw/clone.json&logo=github)](https://github.com/MShawon/github-clone-count-badge)


This code builds upon the NSG repo and incorporates other open-source implementations.

The main branch is the codebase of the RoarGraph paper.
## Getting Started & Reproduce Experiments in the Paper
File format: all `fbin` files begin with the number of vectors (uint32, 4 bytes), dimension (uint32, 4 bytes), and followed by the vector data. (Same format as big-ann competition.)

We use zenodo `https://zenodo.org/` to publish research data and indexes files online (zenodo provides 50GB for free).

0. Prerequisite
```
cmake >= 3.24
g++ >= 9.4
CPU supports AVX-512

Python >= 3.8
Python package:
numpy
urllib
tarfile
```

```
sudo apt install libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
```

```bash
git clone --recursive https://github.com/matchyc/RoarGraph.git
```

1. prepare datasets
The script will download datasets used in the paper and save them in the `./data` directory.
- dataset name:
    - t2i-10M
    - laion-10M
    - webvid-2.5M

Taking the yandex text-to-image dataset as an example.

```bash
bash prepare_data.sh t2i-10M
```

2. Compile and build
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```


3. Bulild Index

3.1 Compute groundtruth for forming a bipartite graph.  
We use a program provided by [here](https://github.com/matchyc/DiskANN/tree/master/tests/utils), which utilizes MKL on the CPU.
You can change the code in the `compute_groundtruth.cpp` file to adjust the memory consumption. (This program will save both vector ids and distances, however, we don't need the later one.)
- base_file: the base data.
- query_file: the training queries.
- gt_file: save path.
- K: $N_q$ in the paper.
```bash
prefix=../data/t2i-10M
cp ./thirdparty/DiskANN/tests/utils/compute_groundtruth compute_groundtruth
mkdir -p ${prefix}
./compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/base.10M.fbin  --query_file ${prefix}/query.train.10M.fbin  --gt_file ${prefix}/train.gt.bin --K 100
```
This step can take hours (see evaluations in Section 5 in the paper). You can just leverage GPU for faster computation, like [raft](https://github.com/rapidsai/raft). It is easy to use by following the instructions, you can reach me out for getting the python scripts to use raft on GPU for these three datasets.

Otherwise, you can slice the training query set and use 10% of it to save much time and evaluate the effects of different training set sizes, wihch can also deliver decent performance.

3.2 build the graph index
- base_data_path: base data path.
- sampled_query_data_path: training queries path.
- projection_index_save_path: where to save the final index.
- learn_base_nn_path: the gt_file computed in 3.1
- M_sq: $N_q$ in the paper.
- M_pjbp: $M$ in the paper.
- L_pjpq: $L$ in the paper.
- T: number of threads employed to construct the graph.
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
prefix=../data/t2i-10M
./tests/test_build_roargraph --data_type float --dist ip \
--base_data_path ${prefix}/base.10M.fbin  \
--sampled_query_data_path ${prefix}/query.train.10M.fbin \
--projection_index_save_path ${prefix}/t2i_10M_roar.index \
--learn_base_nn_path ${prefix}/train.gt.bin \
--M_sq 100 --M_pjbp 35 --L_pjpq 500 -T 64
```

4. Search

- num_threads: number of threads for searching.
- topk: $K$ answers will be returned for evaluation.
- gt_path: the groundtruths for queries in evaluations.
- query_path: queries file for evaluation.
- L_pq: capacities of priority queue during the search phase.
- evaluation_save_path: file path to save performance statistics (optional).

```bash
num_threads=16
topk=10
prefix=../data/t2i-10M
./tests/test_search_roargraph --data_type float \
--dist ip --base_data_path ${prefix}/base.10M.fbin \
--projection_index_save_path ${prefix}/t2i_10M_roar.index \
--gt_path ${prefix}/gt.10k.ibin  \
--query_path ${prefix}/query.10k.fbin \
--L_pq 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
--k ${topk}  -T ${num_threads} \
--evaluation_save_path ${prefix}/test_search_t2i_10M_top${topk}_T${num_threads}.csv
```

## Reproduce the Experiment by Pre-Constructed Indexes
If rigorously reproduction is needed, we provide the constructed indexes for three datasets.
First, download the built indexes used for evaluations.
- https://zenodo.org/records/11090378/files/t2i_10M_roar.index?download=1
- https://zenodo.org/records/11090378/files/laion_10M_roar.index?download=1
- https://zenodo.org/records/11090378/files/webvid_2.5M_roar.index?download=1

Download the query file and ground truth file, set the correct projection_index_save_path as the index path to perform searches.

If you plan to use the constructed index, you can avoid downloading the big dataset files, but only download the query file and ground truth file from links that can be obtained from the `prepare_dataset.sh` script.

## License
MIT License



## Contact
For questions or inquiries, feel free to reach out to Meng Chen at
[mengchen22@m.fudan.edu.cn](mailto:mengchen22@m.fudan.edu.cn)



