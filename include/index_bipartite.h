#include <boost/container/set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "efanna2e/index.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "visited_list_pool.h"

namespace efanna2e {
using LockGuard = std::lock_guard<std::mutex>;
using SharedLockGuard = std::lock_guard<std::shared_mutex>;

class IndexBipartite : public Index {
    typedef std::vector<std::vector<uint32_t>> CompactGraph;

   public:
    explicit IndexBipartite(const size_t dimension, const size_t n, Metric m, Index *initializer);
    virtual ~IndexBipartite();
    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;
    virtual void Search(const float *query, const float *x, size_t k, const Parameters &parameters,
                        unsigned *indices, float *res_dists) override;
    // virtual void BuildBipartite(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
    //                             const Parameters &parameters) override;
    void BuildBipartite(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
                                const Parameters &parameters);

    void BuildRoarGraph(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
                                const Parameters &parameters);
    virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

    inline void SetBipartiteParameters(const Parameters &parameters) {}
    uint32_t SearchBipartiteGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                  unsigned *indices);
    void LinkBipartite(const Parameters &parameters, SimpleNeighbor *simple_graph);
    void LinkOneNode(const Parameters &parameters, uint32_t nid, SimpleNeighbor *simple_graph, bool is_base,
                     boost::dynamic_bitset<> &visited);

    void SearchBipartitebyBase(const float *query, uint32_t nid, const Parameters &parameters,
                               SimpleNeighbor *simple_graph, NeighborPriorityQueue &search_pool,
                               boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset);

    void SearchBipartitebyQuery(const float *query, uint32_t nid, const Parameters &parameters,
                                SimpleNeighbor *simple_graph, NeighborPriorityQueue &search_pool,
                                boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset);

    void LoadVectorData(const char *base_file, const char *sampled_query_file);

    CompactGraph &GetBipartiteGraph() { return bipartite_graph_; }
    inline void InitBipartiteGraph() { bipartite_graph_.resize(total_pts_); }

    inline void LoadSearchNeededData(const char *base_file, const char *sampled_query_file) {
        LoadVectorData(base_file, sampled_query_file);
    }

    void PruneCandidates(std::vector<Neighbor> &search_pool, uint32_t tgt_id, const Parameters &parameters,
                         std::vector<uint32_t> &pruned_list, boost::dynamic_bitset<> &visited);
    void AddReverse(NeighborPriorityQueue &search_pool, uint32_t src_node, std::vector<uint32_t> &pruned_list,
                    const Parameters &parameters, boost::dynamic_bitset<> &visited);

    void BipartiteProjectionReserveSpace(const Parameters &parameters);

    void CalculateProjectionep();

    void LinkProjection(const Parameters &parameters);

    // void LinkBase(const Parameters &parameters, SimpleNeighbor *simple_graph);

    void TrainingLink2Projection(const Parameters &parameters, SimpleNeighbor *simple_graph);

    void SearchProjectionbyQuery(const float *query, const Parameters &parameters, NeighborPriorityQueue &search_pool,
                                 boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset);

    uint32_t PruneProjectionCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                       const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void PruneProjectionBaseSearchCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                             const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void ProjectionAddReverse(uint32_t src_node, const Parameters &parameters);

    void SupplyAddReverse(uint32_t src_node, const Parameters &parameters);

    void PruneProjectionReverseCandidates(uint32_t src_node, const Parameters &parameters,
                                          std::vector<uint32_t> &pruned_list);

    void PruneProjectionInternalReverseCandidates(uint32_t src_node, const Parameters &parameters,
                                                                  std::vector<uint32_t> &pruned_list);

    std::pair<uint32_t, uint32_t> SearchRoarGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                   unsigned *indices, std::vector<float>& res_dists);

    void SaveProjectionGraph(const char *filename);

    void LoadProjectionGraph(const char *filename);

    void LoadNsgGraph(const char *filename);

    void LoadLearnBaseKNN(const char *filename);

    void LoadBaseLearnKNN(const char *filename);

    inline std::vector<std::vector<uint32_t>> &GetProjectionGraph() { return projection_graph_; }

    uint32_t PruneProjectionBipartiteCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                                const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void SearchProjectionGraphInternal(NeighborPriorityQueue &search_queue, const float *query, uint32_t tgt,
                                       const Parameters &parameters, boost::dynamic_bitset<> &visited,
                                       std::vector<Neighbor> &full_retset);

    void PruneBiSearchBaseGetBase(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                  const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void PruneLocalJoinCandidates(uint32_t node, const Parameters &parameters, uint32_t candi);

    void CollectPoints(const Parameters &parameters);

    void dfs(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);

    void InitVisitedListPool(uint32_t num_threads) { visited_list_pool_ = new VisitedListPool(num_threads, nd_); };

    void qbaseNNbipartite(const Parameters &parameters);

    std::pair<uint32_t, uint32_t> SearchBipartiteGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                              unsigned *indices, std::vector<float>& dists);

    Index *initializer_;
    TimeMetric dist_cmp_metric;
    TimeMetric memory_access_metric;
    TimeMetric block_metric;
    VisitedListPool *visited_list_pool_{nullptr};
    bool need_normalize = false;

   protected:
    std::vector<std::vector<uint32_t>> bipartite_graph_;
    std::vector<std::vector<uint32_t>> final_graph_;
    std::vector<std::vector<uint32_t>> projection_graph_;
    std::vector<std::vector<uint32_t>> supply_nbrs_;
    std::vector<std::vector<uint32_t>> learn_base_knn_;
    std::vector<std::vector<uint32_t>> base_learn_knn_;

   private:
    const size_t total_pts_const_;
    size_t total_pts_;
    Distance *l2_distance_;
    // boost::dynamic_bitset<> sq_en_flags_;
    // boost::dynamic_bitset<> bp_en_flags_;
    uint32_t width_;
    std::set<uint32_t> sq_en_set_;
    std::set<uint32_t> bp_en_set_;
    std::mutex sq_set_mutex_;
    std::mutex bp_set_mutex_;
    std::vector<std::mutex> locks_;
    uint32_t u32_nd_;
    uint32_t u32_nd_sq_;
    uint32_t u32_total_pts_;
    uint32_t projection_ep_;
};
}  // namespace efanna2e