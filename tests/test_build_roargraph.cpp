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

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    std::string base_data_file;
    std::string sampled_query_data_file;
    // std::string query_data_file;
    std::string bipartite_index_save_file, projection_index_save_file, learn_base_nn_file, base_learn_nn_file;
    std::string data_type;
    std::string dist;
    uint32_t M_sq;
    uint32_t M_pjbp, L_pjpq;
    // uint32_t L_pq;
    uint32_t num_threads;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("sampled_query_data_path", po::value<std::string>(&sampled_query_data_file)->required(),
                           "Sampled query file in bin format");
        // desc.add_options()("query_data_path",
        //                    po::value<std::string>(&query_data_file)->required(),
        //                    "Query file in bin format");
        // desc.add_options()("bipartite_index_save_path", po::value<std::string>(&bipartite_index_save_file)->required(),
                        //    "Path prefix for saving bipartite index file components");
        desc.add_options()("projection_index_save_path",
                           po::value<std::string>(&projection_index_save_file)->required(),
                           "Path prefix for saving projetion index file components");
        // desc.add_options()("M_bp", po::value<uint32_t>(&M_bp)->default_value(32),
                        //    "Number of neighbors for base points to build the bipartite graph");
        desc.add_options()("M_sq", po::value<uint32_t>(&M_sq)->default_value(32),
                           "Number of neighbors for sampled query points to build the bipartite graph");
        desc.add_options()("M_pjbp", po::value<uint32_t>(&M_pjbp)->default_value(32),
                           "Number of neighbors for projection graph");
        desc.add_options()("L_pjpq", po::value<uint32_t>(&L_pjpq)->default_value(32),
                           "Priority queue length for projection graph searching");

        // desc.add_options()("L_pq", po::value<uint32_t>(&L_pq)->default_value(32),
        //                    "Priority queue length for searching");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        desc.add_options()("learn_base_nn_path", po::value<std::string>(&learn_base_nn_file)->required(),
                           "Path of learn-base NN file");
        // desc.add_options()("base_learn_nn_path", po::value<std::string>(&base_learn_nn_file)->required(),
        //                    "Path of base-learn NN file");

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
    std::cout << "sampled query: " << sampled_query_data_file << std::endl;
    uint32_t base_num, base_dim, sq_num, sq_dim;
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
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

    float *data_bp = nullptr;
    float *data_sq = nullptr;
    float *aligned_data_bp = nullptr;
    float *aligned_data_sq = nullptr;
    efanna2e::Parameters parameters;
    efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
    // efanna2e::load_data<float>(sampled_query_data_file.c_str(), sq_num, sq_dim, data_sq);
    aligned_data_bp = data_bp;
    aligned_data_sq = data_sq;
    // aligned_data_bp = efanna2e::data_align(data_bp, base_num, base_dim);
    // aligned_data_sq = efanna2e::data_align(data_sq, sq_num, sq_dim);

    // if (dist_metric == efanna2e::INNER_PRODUCT) {
    //     efanna2e::ip_normalize(aligned_data_bp, base_dim);
    //     dist_metric = efanna2e::L2;
    // }
    std::cout << "Index save path: " << projection_index_save_file << std::endl;
    efanna2e::IndexBipartite index_bipartite(base_dim, base_num + sq_num, dist_metric, nullptr);
    // parameters.Set<uint32_t>("M_bp", M_bp);
    parameters.Set<uint32_t>("M_sq", M_sq);
    // parameters.Set<uint32_t>("L_pq", L_pq);
    parameters.Set<uint32_t>("M_pjbp", M_pjbp);
    parameters.Set<uint32_t>("L_pjpq", L_pjpq);
    parameters.Set<uint32_t>("num_threads", num_threads);
    // std::cout << "M_bp: " << M_bp << std::endl;
    index_bipartite.LoadLearnBaseKNN(learn_base_nn_file.c_str());
    // index_bipartite.LoadBaseLearnKNN(base_learn_nn_file.c_str());

    omp_set_num_threads(num_threads);
    auto s = std::chrono::high_resolution_clock::now();
    index_bipartite.BuildRoarGraph(sq_num, aligned_data_sq, base_num, aligned_data_bp, parameters);

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";
    index_bipartite.SaveProjectionGraph(projection_index_save_file.c_str());
    std::cout << "Save index to " << projection_index_save_file << std::endl;

    return 0;
}