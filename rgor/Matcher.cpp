/**
 * Copyright (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * RGOR (Relocalization with Generalized Object Recognition) is licensed
 * under Mulan PSL v2. You can use this software according to the terms and
 * conditions of the Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *               http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "Matcher.h"

#include <boost/circular_buffer.hpp>
#include <limits>
#include <nanoflann.hpp>

#include "common/Map.h"
#include "common/macros.h"
#include "utils/KMAssignment.h"
#include "utils/utils.h"

namespace rgor {

template <typename N, typename H>
std::vector<std::tuple<size_t, size_t, float>>
Matcher::operator()(const N &new_lmps, const H &hist_lmps) const {
  using MPKDADT = PointCloudAdaptor<H>;
  using MPKDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, MPKDADT>, MPKDADT, 3>;

  std::vector<std::vector<float>> cost_matrix(
      new_lmps.size(), std::vector<float>(hist_lmps.size(), MATCHER_EPS));
  const MPKDADT kd_cloud(hist_lmps);

  MPKDTree index(3, kd_cloud);
  return Match_<N, H, MPKDTree>(new_lmps, hist_lmps, index, hist_lmps);
}

template <typename N, typename H, typename K>
std::vector<std::tuple<size_t, size_t, float>>
Matcher::operator()(const N &new_lmps, const H &hist_lmps, const K &kdt,
                    const H &kdt_mps) const {
  return Match_<N, H, K>(new_lmps, hist_lmps, kdt, kdt_mps);
}

template <typename N, typename H, typename K>
std::vector<std::tuple<size_t, size_t, float>>
Matcher::Match_(const N &new_lmps, const H &hist_lmps, const K &kdt,
                const H &kdt_mps) const {
  std::vector<std::vector<float>> cost_matrix(
      new_lmps.size(), std::vector<float>(hist_lmps.size(), MATCHER_EPS));
  std::vector<size_t> ret_indexes(knn_search_num_);
  std::vector<float> out_dists_sqr(knn_search_num_, -1);

  if (kdt_mps.size() == 0) {
    return {};
  }
  auto FindNeighbours = [&ret_indexes, &out_dists_sqr, &kdt](size_t num_results,
                                                             float *query_pt) {
    nanoflann::KNNResultSet<float> result_set(num_results);
    result_set.init(&ret_indexes[0], &out_dists_sqr[0]);
    kdt.findNeighbors(result_set, query_pt);
  };

  for (size_t i = 0; i < new_lmps.size(); ++i) {
    if (new_lmps[i] == nullptr) {
      continue;
    }
    float query_pt[3] = {new_lmps[i]->get_pos()[0], new_lmps[i]->get_pos()[1],
                         new_lmps[i]->get_pos()[2]};
    FindNeighbours(knn_search_num_, query_pt);
    for (size_t j = 0; j < knn_search_num_; ++j) {
      if (kdt_mps[ret_indexes[j]] == nullptr) {
        continue;
      }

      size_t idx = std::numeric_limits<size_t>::max();
      if (j < hist_lmps.size() && hist_lmps[j] == kdt_mps[ret_indexes[j]]) {
        idx = j;
      } else {
        for (size_t k = 0; k < hist_lmps.size(); ++k) {
          if (hist_lmps[k] == kdt_mps[ret_indexes[j]]) {
            idx = k;
            break;
          }
        }
        if (idx == std::numeric_limits<size_t>::max()) {
          continue;
        }
      }

      if (hist_lmps[idx] == nullptr) {
        continue;
      }
      auto h_desc = hist_lmps[idx]->get_descriptor();

      auto [ss, sl] = GetScaleMatchScore(new_lmps[i]->get_scale(),
                                         hist_lmps[idx]->get_scale());

      auto scale_score = std::max(ss, sl);

      float dynm_dist_threshold;
      if (ss > sl * 1.5) {
        dynm_dist_threshold =
            dist_scale_ratio_ * new_lmps[i]->get_scale().first;
      } else {
        dynm_dist_threshold =
            dist_scale_ratio_ * new_lmps[i]->get_scale().second;
      }

      dynm_dist_threshold = std::max(dynm_dist_threshold, neighbour_radius_);

      if (out_dists_sqr[j] > dynm_dist_threshold ||
          scale_score < scale_score_threshold_) {
        continue;
      }

      auto n_desc = new_lmps[i]->get_descriptor();
      auto desc_score = 1 - corr_distance(n_desc, h_desc);
      if (desc_score > desc_score_threshold_) {
        cost_matrix[i][idx] = MATCHER_EPS + desc_score +
                              (1 - out_dists_sqr[j] / dynm_dist_threshold) +
                              scale_score;
      }
    }
  }

  auto assignment = Assignment(cost_matrix);
  std::vector<std::tuple<size_t, size_t, float>> ret;
  for (const auto &p : assignment) {
    if (cost_matrix[p.first][p.second] > MATCHER_EPS * 2) {
      ret.emplace_back(p.first, p.second, cost_matrix[p.first][p.second]);
    }
  }
  return ret;
}

template std::vector<std::tuple<size_t, size_t, float>> Matcher::operator()(
    const std::vector<std::shared_ptr<MapPoint<Frame>>> &new_lmps,
    const std::vector<std::shared_ptr<MapPoint<Frame>>> &hist_lmps) const;

template std::vector<std::tuple<size_t, size_t, float>> Matcher::operator()(
    const std::vector<std::shared_ptr<MapPoint<Frame>>> &new_lmps,
    const std::vector<std::shared_ptr<MapPoint<KeyFrame>>> &hist_lmps) const;

template std::vector<std::tuple<size_t, size_t, float>> Matcher::operator()(
    const std::vector<std::shared_ptr<MapPoint<Frame>>> &new_lmps,
    const boost::circular_buffer<std::shared_ptr<MapPoint<Frame>>> &hist_lmps)
    const;

template std::vector<std::tuple<size_t, size_t, float>> Matcher::operator()(
    const std::vector<std::shared_ptr<MapPoint<Frame>>> &new_lmps,
    const std::vector<std::shared_ptr<MapPoint<KeyFrame>>> &hist_lmps,
    const Map::MPKDTree &kdt,
    const std::vector<std::shared_ptr<MapPoint<KeyFrame>>> &kdt_mps) const;

} // namespace rgor