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

#ifndef RGOR_MAPPING_H
#define RGOR_MAPPING_H

#include <atomic>
#include <mutex>
#include <utility>

#include "common/Frame.h"
#include "common/Map.h"
#include "common/Params.h"
#include "toolkits.h"

namespace rgor {

class Mapping {
 public:
  Mapping(Map::Ptr map, const MappingParams &params) {
    if (!map) {
      throw std::invalid_argument("Map pointer cannot be null");
    }
    if (params.dist_threshold < 0 || params.dist_scale_ratio < 0 ||
        params.dist_scale_ratio > 1 || params.desc_score_threshold_1 < 0 ||
        params.desc_score_threshold_1 > 1 ||
        params.scale_score_threshold_1 < 0 ||
        params.scale_score_threshold_1 > 1 ||
        params.desc_score_threshold_2 < 0 ||
        params.desc_score_threshold_2 > 1 ||
        params.scale_score_threshold_2 < 0 || params.scale_score_threshold_2 > 1

    ) {
      throw std::invalid_argument("Parameters not valid");
    }
    map_ = std::move(map);
    dist_threshold_ = params.dist_threshold;
    dist_scale_ratio_ = params.dist_scale_ratio;
    desc_score_threshold_1_ = params.desc_score_threshold_1;
    scale_score_threshold_1_ = params.scale_score_threshold_1;
    desc_score_threshold_2_ = params.desc_score_threshold_2;
    scale_score_threshold_2_ = params.scale_score_threshold_2;
  }

  void AddKeyFrame(KeyFrame::Ptr new_kf);

  std::vector<MapPoint<KeyFrame>::Ptr> GetMPInViews(
      Eigen::Vector4f r_cw, Eigen::Vector3f t_cw, std::shared_ptr<Camera> cam) {
    auto neighbors_mp = map_->GetNeighbors(t_cw, cam->max_depth);
    auto mp_views = MPInViews(neighbors_mp, r_cw, t_cw, cam);
    return mp_views;
  };

  void RemoveMapPoint(MapPoint<KeyFrame>::Ptr mp) { map_->EraseMapPoint(mp); }

  bool get_add_kf_running() const { return add_kf_running_; }

 private:
  Map::Ptr map_;
  std::atomic<bool> add_kf_running_{false};

 private:
  float dist_threshold_ = 0.3;
  float dist_scale_ratio_ = 0.5;
  // XXX 很不优雅，请用更好的描述子
  float desc_score_threshold_1_ = 0.3;
  float scale_score_threshold_1_ = 0.8;
  float desc_score_threshold_2_ = 0.5;
  float scale_score_threshold_2_ = 0.7;
};
}  // namespace rgor

#endif  // RGOR_MAPPING_H
