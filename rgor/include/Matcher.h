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

#ifndef RGOR_MATCHER_H
#define RGOR_MATCHER_H

#include <boost/circular_buffer.hpp>
#include <memory>

#include "common/Frame.h"
#include "common/Map.h"
#include "common/Params.h"

namespace rgor {

struct Matcher {
public:
  Matcher(const MatcherParams &params)
      : knn_search_num_(params.knn_search_num),
        neighbour_radius_(params.neighbour_radius),
        desc_score_threshold_(params.desc_dist_threshold),
        scale_score_threshold_(params.scale_dist_threshold),
        dist_scale_ratio_(params.dist_scale_ratio) {
    if (knn_search_num_ <= 0 || neighbour_radius_ < 0 ||
        desc_score_threshold_ < 0 || desc_score_threshold_ > 1 ||
        scale_score_threshold_ < 0 || scale_score_threshold_ > 1 ||
        dist_scale_ratio_ < 0 || dist_scale_ratio_ > 1) {
      throw std::invalid_argument("Parameters not valid");
    }
  }

  template <typename N, typename H>
  std::vector<std::tuple<size_t, size_t, float>>
  operator()(const N &new_lmps, const H &hist_lmps) const;

  template <typename N, typename H, typename K>
  std::vector<std::tuple<size_t, size_t, float>>
  operator()(const N &new_lmps, const H &hist_lmps, const K &kdt,
             const H &kdt_mps) const;

  template <typename N, typename H, typename K>
  std::vector<std::tuple<size_t, size_t, float>>
  Match_(const N &new_lmps, const H &hist_lmps, const K &kdt,
         const H &kdt_mps) const;

private:
  size_t knn_search_num_;
  float neighbour_radius_;
  float desc_score_threshold_;
  float scale_score_threshold_;
  float dist_scale_ratio_;
};
} // namespace rgor

#endif // RGOR_MATCHER_H
