// index bindings of bipartite_index for python

#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include <unistd.h>

#include "index_bipartite.h"
#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"

namespace py = pybind11;

template <typename T>

class IndexRoarGraph {
   public:
    IndexRoarGraph(const size_t dimension, const size_t n, efanna2e::Metric m) {
        index_ = new efanna2e::IndexBipartite(dimension, n, m, nullptr);
        // init_ = true;
    }

    ~IndexRoarGraph() { delete index_; }


    void setThreads(uint32_t num_threads) {
        omp_set_num_threads(num_threads);
        index_->InitVisitedListPool(2 * num_threads);
    }

    void buildIndex(uint32_t sq_num, uint32_t k_dim, uint32_t base_num, uint32_t M_sq, uint32_t M_pjbp, 
                    uint32_t L_pjpq, uint32_t num_threads, py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &learn_base_nn, py::array_t<float, py::array::c_style | py::array::forcecast> &base_data) {
        efanna2e::Parameters parameters;
        parameters.Set<uint32_t>("M_sq", M_sq);
        parameters.Set<uint32_t>("M_pjbp", M_pjbp);
        parameters.Set<uint32_t>("L_pjpq", L_pjpq);
        parameters.Set<uint32_t>("num_threads", num_threads);
        const uint32_t* learn_base_nn_ptr = &(learn_base_nn.unchecked()(0, 0));
        const float* base_data_ptr = &(base_data.unchecked()(0, 0));
        index_->SetLearnBaseKNN(learn_base_nn_ptr, sq_num, k_dim);
        index_->BuildRoarGraphwithData(sq_num, nullptr, base_num, base_data_ptr, parameters);
    }

    void searchIndex(py::array_t<float, py::array::c_style | py::array::forcecast> &q_data, size_t k, uint32_t L_pq,
                py::array_t<uint32_t, py::array::c_style> &res_id, py::array_t<float, py::array::c_style> &res_dist,
                size_t q_num, uint32_t num_threads, bool using_sq = false) {
        auto items = q_data.unchecked();
        auto res = res_id.mutable_unchecked();
        auto dists = res_dist.mutable_unchecked();
        
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t qid = 0; qid < q_num; qid++) {
                const float *query = &items(qid, 0);
                index_->SearchRoarGraphPy(query, k, qid, L_pq, &res(qid, 0), &dists(qid, 0));
            }
    }

   private:
    efanna2e::IndexBipartite *index_;
    // bool init_ = false;
};
// void Save(std::string filename, efanna2e::IndexBipartite *index_) {
//     index_->SaveProjectionGraph(filename.c_str());
//     index_->SaveReorder(filename);
// }

// void BuildST2(uint32_t M_pjbp, uint32_t L_pjpq, uint32_t num_threads, std::string base_file, efanna2e::Metric m,
//                 std::string traning_set_file, uint32_t each_train_num, uint32_t plan_train_num,
//                 std::string index_save_path) {
//     // if (!init_) {
//     //     std::cout << "Index not initialized" << std::endl;
//     //     return;
//     // }
//     if (traning_set_file.size() == 0) {
//         std::cout << "small dataset" << std::endl;
//         uint32_t base_num, base_dim;

//         efanna2e::load_meta<float>(base_file.c_str(), base_num, base_dim);
//         efanna2e::IndexBipartite * index_ = new efanna2e::IndexBipartite(base_dim, base_num, m, nullptr);
//         float *data_bp = nullptr;
//         efanna2e::load_data_build<float>(base_file.c_str(), base_num, base_dim, data_bp);
//         omp_set_num_threads(num_threads);
//         efanna2e::Parameters parameters;
//         parameters.Set<uint32_t>("M_pjbp", M_pjbp);
//         parameters.Set<uint32_t>("L_pjpq", L_pjpq);
//         parameters.Set<uint32_t>("num_threads", num_threads);
//         index_->BuildGraphOnlyBase(base_num, data_bp, parameters);
//         // index_->LoadProjectionGraph(index_save_path.c_str());
//         std::cout << "Saving data file ..." << std::endl;
//         std::string data_file_name = index_save_path + ".data";
//         index_->SaveBaseData(data_file_name.c_str());
//         std::cout << "Data file saved" << std::endl;
//         std::cout << "roding ..." << std::endl;
//         index_->gorder(index_->gorder_w);
//         std::cout << "roding done" << std::endl;
//         std::cout << "saving graph and order" << std::endl;
//         Save(index_save_path, index_);
//         std::cout << "graph and order saved" << std::endl; 
//         delete index_;
//         malloc_trim(0);
//         std::cout << "create search context for train sq" << std::endl;
//         efanna2e::load_meta<float>(data_file_name.c_str(), base_num, base_dim);
//         uint32_t origin_dim = base_dim;
//         // if (m == efanna2e::INNER_PRODUCT) {
//             uint32_t search_align = DATA_ALIGN_FACTOR;
//             base_dim = (base_dim + search_align - 1) / search_align * search_align;
//         // }
//         index_ = new efanna2e::IndexBipartite(base_dim, base_num, m, nullptr);
//         index_->search_dim_ = origin_dim;
//         index_->LoadProjectionGraph(index_save_path.c_str());
//         std::string order_file_name = std::string(index_save_path) + ".order";
//         std::string original_order_file_name = std::string(index_save_path) + ".original_order";
//         std::cout << "load order..." << std::endl;
//         index_->LoadReorder(order_file_name, original_order_file_name);
//         std::cout << "reorder adjlist..." << std::endl;
//         index_->ReorderAdjList(index_->new_order_);
//         std::cout << "convert to csr..." << std::endl;
//         index_->ConvertAdjList2CSR(index_->row_ptr_, index_->col_idx_, index_->new_order_);
//         std::cout << "load base data in order..." << std::endl;
//         index_->LoadIndexDataReorder(data_file_name.c_str());

//         std::cout << "train sq" << std::endl;
//         index_->InitVisitedListPool(num_threads);    
//         index_->query_file_path = traning_set_file;
//         index_->prefetch_file_path = index_save_path + ".prefetch";
//         index_->quant_file_path = index_save_path + ".quant";
//         index_->TrainQuantizer(index_->get_base_ptr(), base_num, base_dim);
//         //use search 
//         // SearchReorderGraph()
//         index_->FreeBaseData();
//         delete index_;
//         return;
//     }
//     uint32_t base_num, base_dim, sq_num, sq_dim;
//     efanna2e::load_meta<float>(base_file.c_str(), base_num, base_dim);
//     efanna2e::load_meta<float>(traning_set_file.c_str(), sq_num, sq_dim);
//     if (sq_num < plan_train_num) {
//         std::cout << "FAIL: sampled query num is less than plan train num" << std::endl;
//         return;
//     }
//     efanna2e::IndexBipartite * index_ = new efanna2e::IndexBipartite(base_dim, base_num, m, nullptr);

//     index_->train_parts_ = static_cast<size_t>(plan_train_num / each_train_num);
//     index_->each_part_num_ = each_train_num * 1000000;
//     index_->plan_train_num_ = plan_train_num * 1000000;
//     index_->train_data_file = traning_set_file;
//     std::cout << "train_parts: " << index_->train_parts_ << std::endl;
//     float *data_bp = nullptr;
//     efanna2e::load_data_build<float>(base_file.c_str(), base_num, base_dim, data_bp);
//     omp_set_num_threads(num_threads);
//     // index_->LoadLearnBaseKNN(base_file.c_str());
//     efanna2e::Parameters parameters;

//     parameters.Set<uint32_t>("M_pjbp", M_pjbp);
//     parameters.Set<uint32_t>("L_pjpq", L_pjpq);
//     parameters.Set<uint32_t>("num_threads", num_threads);
//     index_->BuildGraphST2(base_num, data_bp, parameters);
//     // index_->LoadProjectionGraph(index_save_path.c_str());
//     std::cout << "Saving data file ..." << std::endl;
//     std::string data_file_name = index_save_path + ".data";
//     index_->SaveBaseData(data_file_name.c_str());
//     std::cout << "Data file saved" << std::endl;
//     std::cout << "roding ..." << std::endl;
//     index_->gorder(index_->gorder_w);
//     std::cout << "roding done" << std::endl;
//     std::cout << "saving graph and order" << std::endl;
//     Save(index_save_path, index_);
//     std::cout << "graph and order saved" << std::endl; 
//     delete index_;
//     malloc_trim(0);
//     std::cout << "create search context for train sq" << std::endl;
//     efanna2e::load_meta<float>(data_file_name.c_str(), base_num, base_dim);
//     uint32_t origin_dim = base_dim;
//     if (m == efanna2e::INNER_PRODUCT) {
//         uint32_t search_align = DATA_ALIGN_FACTOR;
//         base_dim = (base_dim + search_align - 1) / search_align * search_align;
//     }
//     index_ = new efanna2e::IndexBipartite(base_dim, base_num, m, nullptr);
//     index_->search_dim_ = origin_dim;
//     index_->LoadProjectionGraph(index_save_path.c_str());
//     std::string order_file_name = std::string(index_save_path) + ".order";
//     std::string original_order_file_name = std::string(index_save_path) + ".original_order";
//     std::cout << "load order..." << std::endl;
//     index_->LoadReorder(order_file_name, original_order_file_name);
//     std::cout << "reorder adjlist..." << std::endl;
//     index_->ReorderAdjList(index_->new_order_);
//     std::cout << "convert to csr..." << std::endl;
//     index_->ConvertAdjList2CSR(index_->row_ptr_, index_->col_idx_, index_->new_order_);
//     std::cout << "load base data in order..." << std::endl;
//     index_->LoadIndexDataReorder(data_file_name.c_str());

//     std::cout << "train sq" << std::endl;
//     index_->InitVisitedListPool(num_threads);    
//     index_->query_file_path = traning_set_file;
//     index_->prefetch_file_path = index_save_path + ".prefetch";
//     index_->quant_file_path = index_save_path + ".quant";
//     index_->TrainQuantizer(index_->get_base_ptr(), base_num, base_dim);
//     //use search 
//     // SearchReorderGraph()
//     index_->FreeBaseData();
//     delete index_;

// }

// write a load function
// IndexRoarGraph<float> *load(const char *graph_file, const char *data_file, efanna2e::Metric m) {
//     uint32_t base_num, base_dim;
//     efanna2e::load_meta<float>(data_file, base_num, base_dim);
//     IndexRoarGraph<float> *mystery_index = new IndexRoarGraph<float>(base_dim, base_num, m);
//     // mystery_index->Load(graph_file, data_file, m);
//     // mystery_index->LoadwithReorder(graph_file, data_file, m);
//     mystery_index->LoadwithReorderwithSQ(graph_file, data_file, m);
//     // index->LoadProjectionGraph(graph_file);
//     // index->LoadIndexData(data_file);
//     return mystery_index;
// }

PYBIND11_MODULE(RoarGraph, m) {
    m.doc() = "pybind11 RoarGraph plugin";  // optional module docstring
    // enumerate...
    py::enum_<efanna2e::Metric>(m, "Metric")
        .value("L2", efanna2e::Metric::L2)
        .value("IP", efanna2e::Metric::INNER_PRODUCT)
        .value("COSINE", efanna2e::Metric::COSINE)
        .export_values();
    py::class_<IndexRoarGraph<float>>(m, "IndexRoarGraph")
        .def(py::init<const size_t, const size_t, efanna2e::Metric>())
        .def("search", &IndexRoarGraph<float>::searchIndex)
        .def("build", &IndexRoarGraph<float>::buildIndex)
        .def("setThreads", &IndexRoarGraph<float>::setThreads);
}