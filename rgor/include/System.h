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

#ifndef RGOR_SYSTEM_H
#define RGOR_SYSTEM_H

#include <iostream>
#include <memory>
#include <vector>

#include "Mapping.h"
#include "Relocation.h"
#include "Tracking.h"
#include "common/Frame.h"
#include "common/Map.h"
#include "common/Params.h"

namespace rgor {
class System {
public:
  System(const TrackingParams &tracking_params,
         const MatcherParams &matcher_params,
         const MatcherParams &local_matcher_params,
         const MappingParams &mapping_params,
         const RelocationParams &relocation_params,
         std::string_view persistent_map_path = "")
      : map_(std::make_shared<Map>()),
        tracking_(map_, tracking_params, matcher_params, local_matcher_params),
        mapping_(map_, mapping_params), relocation_(relocation_params),
        pmap_(persistent_map_path) {}

  bool AddFrame(Frame &&frame, Eigen::Vector4f &R, Eigen::Vector3f &T);

  void SaveMap(std::string_view path) { map_->DumpMapPoints(path); }

  std::vector<MapPoint<KeyFrame>::Ptr>
  GetMPInViews(Eigen::Vector4f r_cw, Eigen::Vector3f t_cw,
               std::shared_ptr<Camera> cam) {
    return mapping_.GetMPInViews(r_cw, t_cw, cam);
  }

  const Map::Ptr get_map() const { return map_; }

private:
  Map::Ptr map_;

  Tracking tracking_;
  Mapping mapping_;
  Relocation relocation_;

  KeyFrame::Ptr last_kf_;

  PersistentMap pmap_;
};

} // namespace rgor

#endif // RGOR_SYSTEM_H
