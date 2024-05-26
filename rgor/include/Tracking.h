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

#ifndef RGOR_TRACKING_H
#define RGOR_TRACKING_H

#include <boost/circular_buffer.hpp>
#include <unordered_map>

#include "Matcher.h"
#include "common/Frame.h"
#include "common/Params.h"

namespace rgor {

class Tracking {
public:
  explicit Tracking(Map::Ptr map, const TrackingParams &tracking_params,
                    const MatcherParams &map_matcher_params,
                    const MatcherParams &local_matcher_params)
      : map_(std::move(map)), lmps_cache_(tracking_params.lmps_cache_size),
        tmps_cache_(tracking_params.tmps_cache_size),
        obs_threshold_(tracking_params.obs_threshold),
        obs_ratio_threshold_(tracking_params.obs_ratio_threshold),
        nkf_t_threshold_(tracking_params.nkf_t_threshold),
        nkf_r_threshold_(tracking_params.nkf_r_threshold / 180. * M_PI),
        small_object_pixel_(tracking_params.small_object_pixel),
        small_object_scale_(tracking_params.small_object_scale),
        map_matcher_(map_matcher_params), local_matcher_(local_matcher_params) {
    if (obs_threshold_ <= 0 || obs_ratio_threshold_ < 0 ||
        obs_ratio_threshold_ > 1 || nkf_t_threshold_ < 0 ||
        nkf_r_threshold_ < 0 || small_object_pixel_ < 0 ||
        small_object_scale_ < 0) {
      throw std::invalid_argument("Parameters not valid");
    }
  };

  void AddFrame(Frame::Ptr frame);

  bool NeedKeyFrame(Frame::Ptr new_frame, KeyFrame::Ptr last_frame) const;

  KeyFrame::Ptr CreateKeyFrame();

private:
  void LMPFusion(const MapPoint<Frame>::Ptr new_lmp,
                 MapPoint<Frame>::Ptr &matched_lmp) const;

private:
  Map::Ptr map_;
  boost::circular_buffer<std::vector<MapPoint<Frame>::Ptr>> lmps_cache_;
  boost::circular_buffer<MapPoint<Frame>::Ptr> tmps_cache_;

  bool tmps_flag_ = false;

  Frame::Ptr last_frame_;

  std::unordered_map<uuids::uuid, MapPoint<Frame>::Ptr> exist_tmp_map_;

  Matcher map_matcher_;
  Matcher local_matcher_;

private:
  const size_t obs_threshold_;
  const float obs_ratio_threshold_;

  const float nkf_t_threshold_;
  const float nkf_r_threshold_;

  size_t small_object_pixel_;
  float small_object_scale_;
};

} // namespace rgor

#endif // RGOR_TRACKING_H
