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

#ifndef RGOR_COMMON_PARAMS_H
#define RGOR_COMMON_PARAMS_H

namespace rgor {

struct TrackingParams {
  size_t lmps_cache_size = 4;
  size_t tmps_cache_size = 16;
  size_t obs_threshold = 8;
  float obs_ratio_threshold = 0.6;
  float nkf_t_threshold = 0.5;
  float nkf_r_threshold = 0.5;
  size_t small_object_pixel = 5;
  float small_object_scale = 0.1;
};
struct MatcherParams {
  size_t knn_search_num = 8;
  float neighbour_radius = 0.2;
  float desc_dist_threshold = 0.5;
  float scale_dist_threshold = 0.6;
  float dist_scale_ratio = 0.5;
};
struct MappingParams {
  float dist_threshold = 0.2;
  float dist_scale_ratio = 0.7;
  float desc_score_threshold_1 = 0.5;
  float scale_score_threshold_1 = 0.8;
  float desc_score_threshold_2 = 0.5;
  float scale_score_threshold_2 = 0.8;
};
struct RelocationParams {
  float neighbour_radius = 3;
  float scale_score_threshold = 0.8;
  float desc_score_threshold = 0.5;
  float pair_score_threshold = 0.5;
  float best_match_threshold = 2;
  float fine_dist_threshold = 0.3;
  float fine_desc_threshold = 0.3;
};

} // namespace rgor

#endif // RGOR_COMMON_PARAMS_H
