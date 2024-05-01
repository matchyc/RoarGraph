//
// Created by 付聪 on 2017/6/21.
// Modified  by 陈萌 on 2024/4/30
// 

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <xmmintrin.h>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif
namespace efanna2e {

static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

inline float *data_align(float *data_ori, unsigned point_num, unsigned &dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif

    // std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
    float *data_new = 0;
    uint64_t pts = static_cast<uint64_t>(point_num);
    uint64_t d = static_cast<uint64_t>(dim);
    uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
// std::cout << "align to new dim: "<<new_dim << std::endl;
#ifdef __APPLE__
    data_new = new float[new_dim * point_num];
#else
    data_new = (float *)memalign(DATA_ALIGN_FACTOR * 8, pts * new_dim * sizeof(float));
#endif

    for (unsigned i = 0; i < pts; i++) {
        memcpy(data_new + i * new_dim, data_ori + i * d, d * sizeof(float));
        memset(data_new + i * new_dim + d, 0, (new_dim - d) * sizeof(float));
    }
    dim = new_dim;
    std::cout << "new_dim: " << dim << std::endl;
#ifdef __APPLE__
    delete[] data_ori;
#else
    // free(data_ori);
    delete[] data_ori;
#endif
    return data_new;
}

inline void prefetch_vector(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64) _mm_prefetch((const char *)vec + d, _MM_HINT_T0);
}

// load bin meta data from file with different data type, so use template
// get number of points and dimension
template <typename T>
void load_gt_meta(const char *filename, unsigned &points_num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&points_num, 4);
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint32_t calc_contained_pts = (unsigned)((fsize - sizeof(uint32_t) * 2) / (dim) / sizeof(T));
    std::cout << "load gt from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    if (points_num * 2 != calc_contained_pts) {
        std::cerr << "filename: " << std::string(filename) << std::endl;
        std::cerr << "Data file size wrong! Get points " << calc_contained_pts << " but should have " << points_num
                  << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}
template <typename T>
void load_meta(const char *filename, unsigned &points_num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&points_num, 4);
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint32_t calc_contained_pts = (unsigned)((fsize - sizeof(uint32_t) * 2) / (dim) / sizeof(T));
    std::cout << "load meta from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    if (points_num != calc_contained_pts) {
        std::cerr << "filename: " << std::string(filename) << std::endl;
        std::cerr << "Data file size wrong! Get points " << calc_contained_pts << " but should have " << points_num
                  << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T, typename T2>
void load_gt_data_with_dist(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data, T2 *&res_dists) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    data = new T[points_num * dim];
    res_dists = new T2[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * dim), dim * sizeof(T));
    }

    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(res_dists + i * dim), dim * sizeof(T2));
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != points_num * dim * sizeof(T) * 2 + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted!" << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T>
void load_gt_data(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    data = new T[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * dim), dim * sizeof(T));
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != points_num * dim * sizeof(T) + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted!" << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T>
void load_data(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data) {
    std::cout << "load data from file: " << filename << std::endl;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    std::cout << "points_num: " << points_num << " dim: " << dim << std::endl;
    uint64_t pts = static_cast<uint64_t>(points_num);
    uint64_t d = static_cast<uint64_t>(dim);
    uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
    // data = new T[pts * d];
    // check T type

    data = (T*)memalign(DATA_ALIGN_FACTOR * 8, pts * new_dim * sizeof(T));
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * new_dim), d * sizeof(T));
        memset(data + i * new_dim + d, 0, (new_dim - d) * sizeof(T));
        // if ((i + 1) % 100000 == 0)
        //     std::cout << "i: " << i << std::endl;
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != pts * d * sizeof(T) + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted! filename:" << std::string(filename) << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    std::cout << "Finish load data from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    in.close();
}


template<typename T>
inline void normalize(T* arr, const size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    sum += arr[i] * arr[i];
  }
  sum = sqrt(sum);

  for (size_t i = 0; i < dim; i++) {
    arr[i] = (T)(arr[i] / sum);
  }
}
template<typename T>
inline void ip_normalize(T* arr, const size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    sum += arr[i] * arr[i];
  }
//   sum = sqrt(sum);

  for (size_t i = 0; i < dim; i++) {
    arr[i] = (T)(arr[i] / sum);
  }
}

// Metric class to statistic time consuming
class TimeMetric {
   public:
    TimeMetric() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    // return milliseconds
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start_).count();
    }
    // print accumulated time
    void print(const std::string &prompt) { std::cout << prompt << ": " << elapsed_ << "ms" << std::endl; }
    // accumulate elapsed time
    void record() {
        elapsed_ += elapsed();
        // std::cout << prompt << ": " << elapsed_ << "s" << std::endl;
        reset();
    }

   private:
    std::chrono::high_resolution_clock::time_point start_;
    // accumulate elapsed time
    double elapsed_ = 0;
};

}  // namespace efanna2e

#endif  // EFANNA2E_UTIL_H
