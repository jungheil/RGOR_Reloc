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

#include "System.h"

#include <Eigen/src/Geometry/Quaternion.h>

#include <chrono>
#include <memory>
#include <unordered_map>

#include "Relocation.h"
#include "Tracking.h"
#include "include/Mapping.h"

namespace rgor {

std::vector<MapPoint<KeyFrame>::Ptr> GetLocalMapPoint(KeyFrame::Ptr kf,
                                                      KeyFrame::Ptr last_kf) {
  std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr> mps;

  auto init_pose = kf->get_t_cw();

  if (kf != nullptr && !kf->get_bad()) {
    for (auto mp : kf->get_mps()) {
      if (mp != nullptr && !mp->get_bad()) {
        mps[mp->get_uuid()] = mp;
      }
    }
  }
  kf = last_kf;
  while (true) {
    if (kf == nullptr || kf->get_bad()) {
      break;
    }
    if ((kf->get_t_cw() - init_pose).norm() > 5) {
      break;
    }
    for (auto mp : kf->get_mps()) {
      if (mp != nullptr && !mp->get_bad()) {
        mps[mp->get_uuid()] = mp;
      }
    }
    kf = kf->get_parents_kf();
  }

  std::vector<MapPoint<KeyFrame>::Ptr> ret;
  for (auto &item : mps) {
    ret.push_back(item.second);
  }

  return ret;
}

bool System::AddFrame(Frame &&frame, Eigen::Vector4f &R, Eigen::Vector3f &T) {
  R = {0, 0, 0, 0};
  auto frame_ptr = std::make_shared<Frame>(std::move(frame));
  tracking_.AddFrame(frame_ptr);
  bool need_kf = !mapping_.get_add_kf_running() &&
                 tracking_.NeedKeyFrame(frame_ptr, map_->get_last_kf());

  if (need_kf) {
    auto new_kf = tracking_.CreateKeyFrame();

    auto lmps = GetLocalMapPoint(new_kf, last_kf_);

    Map lmap;
    for (auto &mp : lmps) {
      auto m = std::make_shared<MapPoint<KeyFrame>>(
          mp->get_pos(), mp->get_scale(), mp->get_descriptor());
      lmap.AddMapPoint(m);
    }

    mapping_.AddKeyFrame(new_kf);

    if (lmap.get_mps().size() > 0) {
      auto [match, r, t] = relocation_.Get(
          lmap.get_mps(), lmap.get_kd_tree(), lmap.get_mps(), pmap_.get_mps(),
          pmap_.get_kd_tree(), pmap_.get_mps(), pmap_.get_cls_index());
      if (match.size() > 0) {
        auto r_t = new_kf->get_r_cw();
        auto t_t = new_kf->get_t_cw();
        R = r;
        T = t;
      } else {
        R = Eigen::Vector4f(0, 0, 0, -1);
        T = Eigen::Vector3f(0, 0, 0);
      }
    }

    new_kf->set_parents_kf(last_kf_);
    last_kf_ = new_kf;
    return true;
  }
  return false;
}

} // namespace rgor