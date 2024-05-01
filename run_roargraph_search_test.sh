num_threads=16
topk=10

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

prefix=../data/t2i-10M
./tests/test_search_roargraph --data_type float \
--dist ip --base_data_path ${prefix}/base.10M.fbin \
--projection_index_save_path ${prefix}/t2i_10M_roar.index \
--gt_path ${prefix}/gt.10k.ibin  \
--query_path ${prefix}/query.10k.fbin \
--L_pq 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
--k ${topk}  -T ${num_threads} \
--evaluation_save_path ${prefix}/test_search_t2i_10M_top${topk}_T${num_threads}.csv
