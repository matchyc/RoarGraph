#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"

namespace po = boost::program_options;

float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res, uint32_t *gt) {
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<uint32_t> intersection;
        std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
        // check if there duplication in temp_res
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) intersection.push_back(p);
        }

        total_count += static_cast<uint32_t>(intersection.size());
    }
    return static_cast<float>(total_count) / (float)(k * q_num);
}

double ComputeRderr(float* gt_dist, uint32_t gt_dim, std::vector<std::vector<float>>& res_dists, uint32_t k, efanna2e::Metric metric) {
    double total_err = 0;
    uint32_t q_num = res_dists.size();

    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
        std::vector<float> temp_res(res_dists[i].begin(), res_dists[i].end());
        if (metric == efanna2e::INNER_PRODUCT) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = -1.0 * temp_res[j];
            }
        } else if (metric == efanna2e::COSINE) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = 2.0 * ( 1.0 - (-1.0 * temp_res[j]));
            }
        }
        double err = 0.0;
        for (uint32_t j = 0; j < k; j++) {
            err += std::fabs(temp_res[j] - one_gt[j]) / double(one_gt[j]);
        }
        err = err / static_cast<double>(k);
        total_err = total_err + err;
    }
    return total_err / static_cast<double>(q_num);
}

int main(int argc, char **argv) {
    std::string base_data_file;
    std::string query_file;
    std::string sampled_query_data_file;
    std::string gt_file;

    std::string bipartite_index_save_file, projection_index_save_file;
    std::string data_type;
    std::string dist;
    std::vector<uint32_t> L_vec;
    // uint32_t L_pq;
    uint32_t num_threads;
    uint32_t k;
    std::string evaluation_save_path;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("sampled_query_data_path", po::value<std::string>(&sampled_query_data_file)->required(),
                           "Sampled query file in bin format");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(), "Query file in bin format");
        desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(), "Groundtruth file in bin format");
        desc.add_options()("bipartite_index_save_path", po::value<std::string>(&bipartite_index_save_file)->required(),
                           "Path prefix for saving bipartite index file components");
        desc.add_options()("projection_index_save_path",
                           po::value<std::string>(&projection_index_save_file)->required(),
                           "Path prefix for saving projetion index file components");
        desc.add_options()("L_pq", po::value<std::vector<uint32_t>>(&L_vec)->multitoken()->required(),
                           "Priority queue length for searching");
        desc.add_options()("k", po::value<uint32_t>(&k)->default_value(1)->required(), "k nearest neighbors");
        desc.add_options()("evaluation_save_path", po::value<std::string>(&evaluation_save_path),
                           "Path prefix for saving evaluation results");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    uint32_t base_num, base_dim, sq_num, sq_dim;
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);

    efanna2e::Parameters parameters;

    parameters.Set<uint32_t>("num_threads", num_threads);
    omp_set_num_threads(num_threads);
    uint32_t q_pts, q_dim;
    efanna2e::load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float *query_data = nullptr;
    efanna2e::load_data<float>(query_file.c_str(), q_pts, q_dim, query_data);
    float *aligned_query_data = efanna2e::data_align(query_data, q_pts, q_dim);

    uint32_t gt_pts, gt_dim;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    efanna2e::load_gt_meta<uint32_t>(gt_file.c_str(), gt_pts, gt_dim);
    efanna2e::load_gt_data_with_dist<uint32_t, float>(gt_file.c_str(), gt_pts, gt_dim, gt_ids, gt_dists);
    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = efanna2e::INNER_PRODUCT;
        std::cout << "Using inner product as distance metric" << std::endl;
    } else if (dist == "cosine") {
        dist_metric = efanna2e::COSINE;
        std::cout << "Using cosine as distance metric" << std::endl;
    } else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }

    if (!std::filesystem::exists(bipartite_index_save_file.c_str())) {
        std::cout << "bipartite_index_save_file index file does not exist." << std::endl;
        return -1;
    }

    efanna2e::IndexBipartite index(q_dim, base_num + sq_num, dist_metric, nullptr);

    index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str());

    std::cout << "Load graph index: " << bipartite_index_save_file << std::endl;
    index.Load(bipartite_index_save_file.c_str());
    if (index.need_normalize) {
        std::cout << "Normalizing query data" << std::endl;
        for (uint32_t i = 0; i < q_pts; i++) {
            efanna2e::normalize<float>(aligned_query_data + i * q_dim, q_dim);
        }
    }
    index.InitVisitedListPool(num_threads);
    // Search
    std::cout << "k: " << k << std::endl;
    uint32_t *res = new uint32_t[q_pts * k];
    memset(res, 0, sizeof(uint32_t) * q_pts * k);
    std::vector<std::vector<float>> res_dists(q_pts, std::vector<float>(k, 0.0));
    uint32_t *projection_cmps_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(projection_cmps_vec, 0, sizeof(uint32_t) * q_pts);
    uint32_t *hops_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    float *projection_latency_vec = (float *)aligned_alloc(4, sizeof(float) * q_pts);
    memset(projection_latency_vec, 0, sizeof(float) * q_pts);
    std::ofstream evaluation_out;
    if (!evaluation_save_path.empty()) {
        evaluation_out.open(evaluation_save_path, std::ios::out);
    }
    std::cout << "start search" << std::endl;
    for (uint32_t L_pq : L_vec) {
        if (k > L_pq) {
            std::cout << "L_pq must greater or equal than k" << std::endl;
            exit(1);
        }
        parameters.Set<uint32_t>("L_pq", L_pq);
        //pre test
        for (size_t i = 0; i < 100; ++i) {
            index.SearchBipartiteGraph(aligned_query_data + i * q_dim, k, i, parameters, res + i * k, res_dists[i]);
        }

        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < q_pts; ++i) {
            auto ret_val = index.SearchBipartiteGraph(aligned_query_data + i * q_dim, k, i, parameters, res + i * k, res_dists[i]);
            projection_cmps_vec[i] = ret_val.first;
            hops_vec[i] = ret_val.second;
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float qps = (float)q_pts / ((float)diff / 1000.0);
        float recall = ComputeRecall(q_pts, k, gt_dim, res, gt_ids);

        std::cout << "Testing Recall: " << recall << std::endl;
        float avg_projection_cmps = 0.0;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_projection_cmps += projection_cmps_vec[i];
        }
        avg_projection_cmps /= q_pts;

        float avg_hops = 0.0;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_hops += hops_vec[i];
        }
        avg_hops /= (float)q_pts;

        float avg_projection_latency = 0.0;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_projection_latency += projection_latency_vec[i];
        }
        avg_projection_latency /= (float)q_pts;
        double rderr = ComputeRderr(gt_dists, gt_dim, res_dists, k, dist_metric);
        std::cout << L_pq << "\t\t" << qps << "\t\t" << avg_projection_cmps << "\t\t"
                  << ((float)diff / q_pts) << "\t\t" << recall << "\t\t" << rderr << "\t\t" << avg_hops << std::endl;
        if (evaluation_out.is_open()) {
            evaluation_out << L_pq << "," << qps << "," << avg_projection_cmps << "," << ((float)diff / q_pts) << ","
                           << recall << "," << rderr << "," << avg_hops << std::endl;
        }
    }
    if (evaluation_out.is_open()) {
        evaluation_out.close();
    }

    delete[] res;
    free(projection_cmps_vec);
    free(hops_vec);
    free(projection_latency_vec);
    delete[] aligned_query_data;
    delete[] gt_ids;

    return 0;
}