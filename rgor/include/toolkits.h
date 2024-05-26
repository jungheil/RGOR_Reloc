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
#ifndef RGOR_TOOLKITS_H_
#define RGOR_TOOLKITS_H_

#include <vector>

#include "common/Frame.h"
#include "common/Map.h"

namespace rgor {

template <typename T>
T MPInViews(T &mps, Eigen::Vector4f r_cw, Eigen::Vector3f t_cw,
            std::shared_ptr<Camera> cam) {
  T ret;
  Eigen::Quaternionf q(r_cw);
  Eigen::Matrix3f R_wc = q.toRotationMatrix().transpose();

  for (auto &mp : mps) {
    if (mp == nullptr || !mp->get_on_map() || mp->get_bad()) {
      continue;
    }
    Eigen::Vector3f pos = mp->get_pos();
    if ((pos - t_cw).norm() > cam->max_depth ||
        (pos - t_cw).norm() < cam->min_depth) {
      continue;
    }
    Eigen::Vector3f pos_c = R_wc * (pos - t_cw);
    if (pos_c(2) <= 0) {
      continue;
    }
    Eigen::Vector3f pos_img = cam->K * pos_c / pos_c(2);
    float scale_img = mp->get_scale().second * cam->K(0, 0) / pos_c(2);
    if (pos_img(0) - scale_img / 2 <
            cam->width * (1 - cam->valid_wh_ratio) / 2 ||
        pos_img(0) + scale_img / 2 >
            cam->width * (1 + cam->valid_wh_ratio) / 2 ||
        pos_img(1) - scale_img / 2 <
            cam->height * (1 - cam->valid_wh_ratio) / 2 ||
        pos_img(1) + scale_img / 2 >
            cam->height * (1 + cam->valid_wh_ratio) / 2) {
      continue;
    }

    ret.emplace_back(mp);
  }
  return ret;
}
} // namespace rgor
#endif // RGOR_TOOLKITS_H_
