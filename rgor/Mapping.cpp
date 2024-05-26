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

#include "Mapping.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cstddef>
#include <thread>

#include "Optimizer.h"
#include "common/Frame.h"
#include "common/Map.h"
#include "utils/utils.h"

namespace rgor {

void Mapping::AddKeyFrame(KeyFrame::Ptr new_kf) {
  if (add_kf_running_) {
    return;
  }
  new_kf->set_on_map(true);
  map_->AddKeyFrame(new_kf);

  for (const auto mp : new_kf->get_mps()) {
    if (mp != nullptr && !mp->get_on_map()) {
      auto mp_views = GetMPInViews(new_kf->get_r_cw(), new_kf->get_t_cw(),
                                   new_kf->get_camera());

      std::vector<MapPoint<KeyFrame>::Ptr> mp_iou_neighbors;
      size_t best_iou_idx = 0;
      size_t max_iou_score = 0;

      Eigen::Quaternionf q(new_kf->get_r_cw());
      Eigen::Matrix3f R_wc = q.toRotationMatrix().transpose();
      auto mp_c = R_wc * (mp->get_pos() - new_kf->get_t_cw());
      auto mp_i = new_kf->get_camera()->K * mp_c;
      auto mp_scale = mp->get_scale().first *
                      (mp->get_pos() - new_kf->get_t_cw()).norm() /
                      new_kf->get_camera()->K(0, 0);
      auto mp_img = new_kf->get_camera()->K;
      for (const auto &mpv : mp_views) {
        if (mpv->get_bad() || !mpv->get_on_map()) {
          continue;
        }
        if ((mpv->get_pos() - mp->get_pos()).norm() > 2) {
          continue;
        }
        auto mpv_c = R_wc * (mpv->get_pos() - new_kf->get_t_cw());
        auto mpv_i = new_kf->get_camera()->K * mpv_c;
        auto mpv_scale = mpv->get_scale().first *
                         (mpv->get_pos() - new_kf->get_t_cw()).norm() /
                         new_kf->get_camera()->K(0, 0);
        auto scale_score =
            std::min(mp_scale, mpv_scale - (mp_i - mpv_i).norm()) /
            std::max(mp_scale, mpv_scale);
        auto desc_score =
            1 - corr_distance(mp->get_descriptor(), mpv->get_descriptor());
        if (scale_score > scale_score_threshold_1_ &&
                desc_score > desc_score_threshold_1_ ||
            desc_score > desc_score_threshold_2_ &&
                scale_score > scale_score_threshold_2_) {
          mp_iou_neighbors.push_back(mpv);
          if (scale_score > max_iou_score) {
            max_iou_score = scale_score;
            best_iou_idx = mp_iou_neighbors.size() - 1;
          }
        }
      }
      if (!mp_iou_neighbors.empty()) {
        MapPoint<KeyFrame>::Ptr best_mpv = mp_iou_neighbors[best_iou_idx];
        for (auto &obs : mp->get_observations()) {
          if (auto kf = obs.second.first.lock()) {
            best_mpv->AddObservation(kf, obs.second.second);
          }
        }
        std::vector<Eigen::VectorXf> descs;
        std::vector<std::pair<float, float>> scales;
        for (const auto &obs : best_mpv->get_observations()) {
          if (auto kf = obs.second.first.lock()) {
            if (kf->get_bad() || !kf->get_on_map()) {
              continue;
            }
            descs.push_back(
                kf->get_measurement()[obs.second.second].descriptor);
            scales.push_back(kf->get_measurement()[obs.second.second].scale);
          }
        }
        if (!descs.empty()) {
          auto desc = BestDescriptor(descs);
          best_mpv->SetDescriptor(desc);
          auto scale = BestScale(scales);
          best_mpv->set_scale(scale);
        }
        for (size_t i = 0; i < mp_iou_neighbors.size(); ++i) {
          if (i != best_iou_idx) {
            RemoveMapPoint(mp_iou_neighbors[i]);
          }
        }
        continue;
      }

      std::vector<MapPoint<KeyFrame>::Ptr> mp_neighbors;
      size_t best_neighbor_idx = 0;
      size_t max_score = 0;
      for (const auto &mpv : mp_views) {
        if (mpv->get_bad() || !mpv->get_on_map()) {
          continue;
        }
        auto dist = (mp->get_pos() - mpv->get_pos()).norm();

        // XXX 到底要不要平方
        float dynm_dist_threshold = std::max(
            dist_threshold_, mpv->get_scale().first * dist_scale_ratio_);

        auto desc_score =
            1 - corr_distance(mp->get_descriptor(), mpv->get_descriptor());
        auto scale_score = GetScaleScore(mp->get_scale(), mpv->get_scale());

        if (dist < dynm_dist_threshold &&
                desc_score > desc_score_threshold_1_ &&
                scale_score > scale_score_threshold_1_ ||
            dist < dynm_dist_threshold &&
                desc_score > desc_score_threshold_2_ &&
                scale_score > scale_score_threshold_2_) {
          mp_neighbors.emplace_back(mpv);
          if (scale_score + desc_score > max_score) {
            max_score = scale_score + desc_score;
            best_neighbor_idx = mp_neighbors.size() - 1;
          }
        }
      }
      if (mp_neighbors.empty()) {
        map_->AddMapPoint(mp);
      } else {
        // TODO
        // 这里可以考虑把所有else的都融合,一个地区只能有一个点,可以直接把邻近但不相似忽略掉
        MapPoint<KeyFrame>::Ptr best_mpv = mp_neighbors[best_neighbor_idx];
        for (auto &obs : mp->get_observations()) {
          if (auto kf = obs.second.first.lock()) {
            best_mpv->AddObservation(kf, obs.second.second);
          }
        }
        std::vector<Eigen::VectorXf> descs;
        std::vector<std::pair<float, float>> scales;
        for (const auto &obs : best_mpv->get_observations()) {
          if (auto kf = obs.second.first.lock()) {
            if (kf->get_bad() || !kf->get_on_map()) {
              continue;
            }
            descs.push_back(
                kf->get_measurement()[obs.second.second].descriptor);
            scales.push_back(kf->get_measurement()[obs.second.second].scale);
          }
        }
        if (!descs.empty()) {
          auto desc = BestDescriptor(descs);
          best_mpv->SetDescriptor(desc);
          auto scale = BestScale(scales);
          best_mpv->set_scale(scale);
        }
        for (size_t i = 0; i < mp_neighbors.size(); ++i) {
          if (i != best_neighbor_idx) {
            RemoveMapPoint(mp_neighbors[i]);
          }
        }
      }
    } else {
      if (mp != nullptr) {
        std::vector<Eigen::VectorXf> dists;
        std::vector<std::pair<float, float>> scales;
        for (const auto &obs : mp->get_observations()) {
          if (auto kf = obs.second.first.lock()) {
            dists.push_back(
                kf->get_measurement()[obs.second.second].descriptor);
            scales.push_back(kf->get_measurement()[obs.second.second].scale);
          }
        }
        if (!dists.empty()) {
          auto desc = BestDescriptor(dists);
          auto old_cls = mp->get_cls();
          mp->SetDescriptor(desc);
          map_->UpdateClsIndex(mp, old_cls);
          auto scale = BestScale(scales);
          mp->set_scale(scale);
        }
      }
    }
  }

  add_kf_running_ = true;
  std::shared_ptr<std::thread> opt_thread_ =
      std::make_shared<std::thread>([this, new_kf]() {
        LocalBundleAdjustment(new_kf, this->map_);
        this->add_kf_running_ = false;
      });
  opt_thread_->detach();
};

} // namespace rgor