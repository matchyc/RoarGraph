# RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search | MysteryANN

This repository includes the codes for VLDB 2024 paper RoarGraph, it also 🏆 Winning NeurIPS' Competition Track: Big ANN, Practical Vector Search Challenge. (OOD Track) (Our other solution won the Sparse Track).

[![NIPS Big-ANN Benchmark 2023](https://img.shields.io/badge/NIPS%20Big--ANN%20Benchmark-2023-blue)](https://big-ann-benchmarks.com/neurips23.html)

![](https://api.visitorbadge.io/api/VisitorHit?user=matchyc&repo=mysteryann&countColor=%237B1E7A)



[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/matchyc/daf1f1c1372416a529003f91b5562fdc/raw/clone.json&logo=github)](https://github.com/MShawon/github-clone-count-badge)


This code builds upon the NSG repo and incorporates other open-source implementations.

The main branch is the codebase of the RoarGraph paper.
The codes for the NIPS 2023 challenge are available in separate branches.

## Getting Started & Reproduce Experiments in the Paper
File format: all `fbin` files begin with number of vectors (uint32, 4 bytes), dimension (uint32, 4 bytes), and followed by the vector data.

We use zenodo `https://zenodo.org/` to save indexes files online (50GB for free), however, it may take a while to download file with x GB size (tested 500KB/s) since its a free platform for publishing research data.

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

1. prepare datasets
The script will download datasets used in the paper and save them in the `./data` directory.
- dataset name:
    - t2i-10M
    - LAION-10M
    - WebVid-2.5M
```bash
bash prepare_data.sh <dataset name>
```

2. Compile and build
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```


3. Bulild Index

3.1 Compute groundtruth for forming a bipartite graph
We use program provided by [here](https://github.com/matchyc/DiskANN/tree/master/tests/utils), which utilizes MKL on CPU.
You can change the code in the `compute_groundtruth.cpp` file to adjust the memory comsumption. (This program will save both vector ids and distances, we don't need the later one.)
- base_file: the base data.
- query_file: the training queries.
- gt_file: save path.
```bash
prefix=../data/t2i-10M
./compute_groundtruth --data_type float --dist_fn l2  --base_file ${prefix}/base.10M.fbin  --query_file ${prefix}/query.train.10M.fbin  --gt_file ${prefix}/train.gt.bin --K 100
```
However, it can take hours (see Section 5 in the paper). You can just leverage GPU for faster computation, like [raft](https://github.com/rapidsai/raft). It is easy to use by following the instructions, you can reach me out for getting the python scripts to use raft on GPU. Otherwise, you can slice the training query file and use 10% of it, wihch can also deliver decent performance.

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
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
prefix=../data/t2i-10M
./tests/test_build_roargraph --data_type float --dist ip \
--base_data_path ${prefix}/base.10M.fbin  \
--sampled_query_data_path ${prefix}/query.train.10M.fbin \
--projection_index_save_path ${prefix}/t2i_10M_roar.index \
--learn_base_nn_path ${prefix}/t2i.train.in.base.nn.dist.10M.ibin \
--M_sq 100 --M_pjbp 35 --L_pjpq 500 -T 64
```

4. Search

- num_threads: number of threads for searching.
- topk: $K$ answers will be returned for evaluation.
- gt_path: the groundtruths for queries in evaluations.
- query_path: queries file for evalution.
- L_pq: capacities of priority queue during the search phase.
- evaluation_save_path: file path to save performance statistics (optional).
```bash
num_threads=16
topk=10
prefix=../data/t2i-10M
./tests/test_search_roargraph --data_type float \
--dist ip --base_data_path ${prefix}/base.10M.fbin \
--projection_index_save_path ${prefix}/t2i_10M_roar.index \
--gt_path ${prefix}/groundtruth.base.10M.query.10k.ibin \
--query_path ${prefix}/query.public.10k.fbin \
--L_pq 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
--k ${topk}  -T ${num_threads} \
--evaluation_save_path ${prefix}/test_search_t2i_10M_top${topk}_T${num_threads}.csv
```

## Reproduce the Experiment
If rigorously reproduce is needed, we provide the constructed indexes for three datasets.
First, download them and then search on these indexes can reproduce the performance experiment in the paper.
- https://zenodo.org/records/11073098/files/t2i_10M_roar.index?download=1
- https://zenodo.org/records/11073098/files/laion_10M_roar.index?download=1
- https://zenodo.org/records/11073098/files/webvid_2.5M_roar.index?download=1

Simply download the index and set the projection_index_save_path as the index path to perform searches. If downloading takes too long, you can request that I upload/send the index files to you for strict reproduction, provided you can offer a suitable file sharing platform.

## License
MIT License



## Contact
If you wish, please leave a message if you plan to use this idea.

For questions or inquiries, feel free to reach out to Meng Chen at
[mengchen22@m.fudan.edu.cn](mailto:mengchen22@m.fudan.edu.cn)
<!-- [mengchen9909@gmail.com](mailto:mengchen9909@gmail.com) -->



