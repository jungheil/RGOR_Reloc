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

#include "Tracking.h"

#include <Eigen/src/Geometry/Quaternion.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "common/Frame.h"
#include "utils/utils.h"

namespace rgor {
void Tracking::AddFrame(Frame::Ptr frame) {
  std::vector<MapPoint<Frame>::Ptr> lmps;
  for (size_t i = 0; i < frame->get_kps().size(); ++i) {
    Eigen::Vector3f pt = frame->get_kp_position()[i];
    if (pt == Eigen::Vector3f::Zero()) {
      continue;
    }
    auto scale = EstimateScale<float>(
        frame->get_kps()[i]->w, frame->get_kps()[i]->h,
        frame->get_kps()[i]->depth * 1e-3, frame->get_camera()->K(0, 0),
        frame->get_camera()->K(1, 1));
    Eigen::Vector3f v = pt - frame->get_t_cw();
    if (v.norm() > 1e-6) {
      v.normalize();
    }
    v = v * (scale.second / 2);
    pt = pt + v;

    if (pt != Eigen::Vector3f::Zero()) {
      auto lmp = std::make_shared<MapPoint<Frame>>(
          pt, scale, frame->get_kps()[i]->descriptor);
      if (std::max(frame->get_kps()[i]->w, frame->get_kps()[i]->h) <
              small_object_pixel_ ||
          lmp->get_scale().first < small_object_scale_) {
        continue;
      }
      lmp->AddObservation(frame, i);
      lmps.emplace_back(std::move(lmp));
    }
  }

  // Match new map points with existing map points
  std::vector<std::tuple<size_t, size_t, float>> map_match =
      map_matcher_(lmps, map_->get_mps(), map_->get_kd_tree(), map_->get_mps());
  for (auto &match : map_match) {
    auto [new_idx, hist_idx, dist] = match;
    if (lmps[new_idx] == nullptr) {
      continue;
    }
    if (exist_tmp_map_.find(map_->get_mps()[hist_idx]->get_uuid()) ==
        exist_tmp_map_.end()) {
      exist_tmp_map_[map_->get_mps()[hist_idx]->get_uuid()] = lmps[new_idx];
    } else {
      LMPFusion(lmps[new_idx],
                exist_tmp_map_[map_->get_mps()[hist_idx]->get_uuid()]);
    }
    exist_tmp_map_[map_->get_mps()[hist_idx]->get_uuid()]
        ->IncreaseRealObservedTimes();
    lmps[new_idx] = nullptr;
  }
  for (auto mp : exist_tmp_map_) {
    mp.second->IncreaseObservedTimes();
  }

  // Match new map points with on tracking local map points
  auto tracked_match = local_matcher_(lmps, tmps_cache_);
  for (auto &match : tracked_match) {
    auto [new_idx, hist_idx, dist] = match;
    if (lmps[new_idx] == nullptr || tmps_cache_[hist_idx] == nullptr) {
      std::cout << "Tracking::AddFrame: lmps[new_idx] or tmps_cache_[hist_idx] "
                   "is nullptr"
                << std::endl;
      continue;
    }
    LMPFusion(lmps[new_idx], tmps_cache_[hist_idx]);
    if (tmps_cache_[hist_idx]->get_observations().size() > obs_threshold_ &&
        tmps_cache_[hist_idx]->GetObservedRatio() > obs_ratio_threshold_) {
      tmps_flag_ = true;
    }
    tmps_cache_[hist_idx]->IncreaseRealObservedTimes();
    lmps[new_idx] = nullptr;
  }
  for (auto mp : tmps_cache_) {
    if (mp != nullptr) {
      mp->IncreaseObservedTimes();
    }
  }

  // Match new map points with historical points
  for (size_t i = lmps_cache_.size(); i > 0; --i) {
    std::vector<MapPoint<Frame>::Ptr> hist_lmps = lmps_cache_[i - 1];
    if (hist_lmps.empty()) {
      continue;
    }
    auto hist_match = local_matcher_(lmps, hist_lmps);

    for (auto &match : hist_match) {
      auto [new_idx, hist_idx, dist] = match;
      if (lmps[new_idx] == nullptr || hist_lmps[hist_idx] == nullptr) {
        continue;
      }
      LMPFusion(lmps[new_idx], hist_lmps[hist_idx]);
      hist_lmps[hist_idx]->IncreaseObservedTimes();
      hist_lmps[hist_idx]->IncreaseRealObservedTimes();
      tmps_cache_.push_back(hist_lmps[hist_idx]);
      lmps_cache_[i - 1][hist_idx] = nullptr;
      lmps[new_idx] = nullptr;
    }
  }

  last_frame_ = frame;
  lmps_cache_.push_back(std::move(lmps));
};

bool Tracking::NeedKeyFrame(Frame::Ptr new_frame,
                            KeyFrame::Ptr last_frame) const {
  if (last_frame == nullptr) {
    return true;
  }
  if (tmps_flag_) {
    return true;
  }
  // if (last_frame->get_timestamp() - new_frame->get_timestamp() >
  //            kf_timeout_threshold_) {
  //   return true;
  // }
  auto delta_t = new_frame->get_t_cw() - last_frame->get_t_cw();
  auto q_n = Eigen::Quaternionf(new_frame->get_r_cw());
  auto q_l = Eigen::Quaternionf(last_frame->get_r_cw());
  auto q = q_l.conjugate() * q_n;
  auto delta_q = 2 * acos(q.w());
  if (delta_q > nkf_r_threshold_ || delta_t.norm() > nkf_t_threshold_) {
    return true;
  }
  return false;

  // NOTE 是否还需要根据关键帧被跟踪特征点丢失个数判断
};

KeyFrame::Ptr Tracking::CreateKeyFrame() {
  // count tmps_cache_ observations in each frame
  std::unordered_map<uuids::uuid, std::pair<Frame::Ptr, size_t>>
      frame_obs_count;
  for (auto &lmp : tmps_cache_) {
    if (lmp == nullptr) {
      continue;
    }
    for (auto &obs : lmp->get_observations()) {
      frame_obs_count[obs.first].second++;
    }
  }
  // find the frame with the most observations, if same count, choose the
  // latest one
  Frame::Ptr base_frame = last_frame_;

  size_t max_obs = -1;
  time_t last_frame_time = 0;

  for (auto &obs : frame_obs_count) {
    if (obs.second.second > max_obs) {
      max_obs = obs.second.second;
      base_frame = obs.second.first;
    } else if (obs.second.second == max_obs) {
      if (obs.second.first->get_timestamp() > last_frame_time) {
        last_frame_time = obs.second.first->get_timestamp();
        base_frame = obs.second.first;
      }
    }
  }

  // create keyframe
  assert(base_frame != nullptr);

  std::vector<KFMeasurement> measurement;
  measurement.reserve(exist_tmp_map_.size());
  std::vector<std::shared_ptr<MapPoint<KeyFrame>>> mps;
  // 添加已存在地图点的观测与mp

  for (auto &mp : exist_tmp_map_) {
    if (mp.second->GetObservedRatio() < obs_ratio_threshold_ &&
        mp.second->get_recent_observed_times() == 0) {
      continue;
    } else {
      mp.second->ClearRecentObservedTimes();
    }

    // world to cam
    auto R = Eigen::Quaternionf(base_frame->get_r_cw()).toRotationMatrix();
    auto t = base_frame->get_t_cw();
    auto meas = R.transpose() * (mp.second->get_pos() - t);
    measurement.push_back(
        {meas, mp.second->get_descriptor(), mp.second->get_scale()});
    mps.push_back(map_->GetMapPointPtr(mp.first));
  }

  for (auto itr = tmps_cache_.begin(); itr != tmps_cache_.end(); ++itr) {
    if (*itr != nullptr && (*itr)->get_observations().size() > obs_threshold_ &&
        (*itr)->GetObservedRatio() > obs_ratio_threshold_) {
      auto R = Eigen::Quaternionf(base_frame->get_r_cw()).toRotationMatrix();
      auto t = base_frame->get_t_cw();
      auto meas = R.transpose() * ((*itr)->get_pos() - t);
      measurement.push_back(
          {meas, (*itr)->get_descriptor(), (*itr)->get_scale()});
      mps.emplace_back(std::make_shared<MapPoint<KeyFrame>>(
          (*itr)->get_pos(), (*itr)->get_scale(), (*itr)->get_descriptor()));
      *itr = nullptr;
    }
  }
  tmps_flag_ = false;
  exist_tmp_map_.clear();

  auto new_kf = std::make_shared<KeyFrame>(
      base_frame->get_camera(), base_frame->get_r_cw(), base_frame->get_t_cw(),
      base_frame->get_timestamp(), mps, measurement);
  for (size_t i = 0; i < new_kf->get_mps().size(); ++i) {
    if (new_kf->get_mps()[i] == nullptr) {
      continue;
    }
    new_kf->get_mps()[i]->AddObservation(new_kf, i);
  }

  return new_kf;
};

void Tracking::LMPFusion(const MapPoint<Frame>::Ptr new_lmp,
                         MapPoint<Frame>::Ptr &matched_lmp) const {
  if (!new_lmp || !matched_lmp) {
    std::cout << "LMPFusion: new_lmp or matched_lmp is nullptr" << std::endl;
    return;
  }

  //        // 保存实验数据
  //        std::ofstream outfile("~/rgor_data/descriptors.txt");
  //        auto nd = new_lmp->get_descriptor();
  //        outfile << nd.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ",",
  //        ",", "", "", "", "")) << std::endl; outfile.close();

  for (const auto &obs : new_lmp->get_observations()) {
    if (auto frame = obs.second.first.lock()) {
      matched_lmp->AddObservation(frame, obs.second.second);
    }
  }
  // for (auto itr = new_lmp->get_observations().begin();
  //      itr != new_lmp->get_observations().end(); ++itr) {
  //   if (auto frame = itr->second.first.lock()) {
  //     matched_lmp->AddObservation(frame, itr->second.second);
  //   }
  // }
  std::vector<Eigen::VectorXf> descs;
  descs.reserve(matched_lmp->get_observations().size());
  for (auto &obs : matched_lmp->get_observations()) {
    if (auto frame = obs.second.first.lock()) {
      descs.push_back(frame->get_kps()[obs.second.second]->descriptor);
    }
  }
  auto desc = BestDescriptor(descs);
  if (desc.size() == 0) {
    throw std::runtime_error("Descriptor is not initialized");
  }
  matched_lmp->SetDescriptor(desc);

  // TODO Bayes filter
  auto pos_new = new_lmp->get_pos();
  auto pos_mc = matched_lmp->get_pos();
  matched_lmp->set_pos((pos_new + pos_mc) / 2);
  auto scale_new = new_lmp->get_scale();
  auto scale_mc = matched_lmp->get_scale();
  matched_lmp->set_scale(
      std::make_pair((scale_new.first + scale_mc.first) / 2.,
                     (scale_new.second + scale_mc.second) / 2.));
};

} // namespace rgor