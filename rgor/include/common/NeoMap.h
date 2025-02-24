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

#ifndef RGOR_NEO_MAP_H
#define RGOR_NEO_MAP_H

#include <Eigen/Core>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "nanoflann.hpp"
#include "utils/FeatureDB.h"
#include "utils/utils.h"
#include "uuid.h"

// 新的类数据应该和方法解耦

namespace rgor {

class NeoMapPoint {
 public:
  using Ptr = std::shared_ptr<NeoMapPoint>;

 public:
  NeoMapPoint(uuids::uuid uuid, Eigen::VectorXf descriptor, Eigen::Vector3f pos,
              std::pair<float, float> scale,
              std::unordered_set<uuids::uuid> observations)
      : uuid_(uuid),
        descriptor_(descriptor),
        pos_(pos),
        scale_(scale),
        observations_(observations) {
    updated_at_ = std::chrono::system_clock::now();
    created_at_ = std::chrono::system_clock::now();
  }
  const uuids::uuid get_uuid() {
    std::shared_lock lock(mutex_);
    return uuid_;
  }

  // void set_uuid(const uuids::uuid &uuid) {
  //   std::unique_lock lock(mutex_);
  //   uuid_ = uuid;
  // }
  const Eigen::VectorXf get_descriptor() {
    std::shared_lock lock(mutex_);
    return descriptor_;
  }

  void set_descriptor(const Eigen::VectorXf &descriptor) {
    std::unique_lock lock(mutex_);
    descriptor_ = descriptor;
    updated_at_ = std::chrono::system_clock::now();
  }
  const Eigen::Vector3f get_pos() {
    std::shared_lock lock(mutex_);
    return pos_;
  }

  void set_pos(const Eigen::Vector3f &pos) {
    std::unique_lock lock(mutex_);
    pos_ = pos;
    updated_at_ = std::chrono::system_clock::now();
  }
  const std::pair<float, float> get_scale() {
    std::shared_lock lock(mutex_);
    return scale_;
  }

  void set_scale(const std::pair<float, float> &scale) {
    std::unique_lock lock(mutex_);
    scale_ = scale;
    updated_at_ = std::chrono::system_clock::now();
  }
  const std::unordered_set<uuids::uuid> get_observations() {
    std::shared_lock lock(mutex_);
    return observations_;
  }

  void set_observations(const std::unordered_set<uuids::uuid> &observations) {
    std::unique_lock lock(mutex_);
    observations_ = observations;
    updated_at_ = std::chrono::system_clock::now();
  }

  const std::chrono::system_clock::time_point get_updated_at() {
    std::shared_lock lock(mutex_);
    return updated_at_;
  }

  void set_updated_at(const std::chrono::system_clock::time_point &updated_at) {
    std::unique_lock lock(mutex_);
    updated_at_ = updated_at;
    updated_at_ = std::chrono::system_clock::now();
  }

  const std::chrono::system_clock::time_point get_created_at() {
    std::shared_lock lock(mutex_);
    return created_at_;
  }

  void set_created_at(const std::chrono::system_clock::time_point &created_at) {
    std::unique_lock lock(mutex_);
    created_at_ = created_at;
    updated_at_ = std::chrono::system_clock::now();
  }

 private:
  uuids::uuid uuid_;
  Eigen::VectorXf descriptor_;
  std::unordered_set<uuids::uuid> observations_;
  alignas(64) std::pair<float, float> scale_;
  Eigen::Vector3f pos_;

  std::chrono::system_clock::time_point updated_at_ =
      std::chrono::system_clock::now();
  std::chrono::system_clock::time_point created_at_ =
      std::chrono::system_clock::now();
  std::shared_mutex mutex_;
};

class NeoKeyFrame {
 public:
  using Ptr = std::shared_ptr<NeoKeyFrame>;

 public:
  NeoKeyFrame(uuids::uuid uuid, Eigen::Vector4f rel_r_cw,
              Eigen::Vector3f rel_t_cw, Eigen::Vector4f abs_r_cw,
              Eigen::Vector3f abs_t_cw,
              const std::unordered_map<uuids::uuid, Eigen::Vector3f> mps,
              Ptr pre_kf)
      : uuid_(uuid),
        rel_r_cw_(rel_r_cw),
        rel_t_cw_(rel_t_cw),
        abs_r_cw_(abs_r_cw),
        abs_t_cw_(abs_t_cw),
        measurement_mps_(mps) {
    updated_at_ = std::chrono::system_clock::now();
    created_at_ = std::chrono::system_clock::now();
  }

  const uuids::uuid get_uuid() {
    std::shared_lock lock(mutex_);
    return uuid_;
  }

  const Eigen::Vector4f get_rel_r_cw() {
    std::shared_lock lock(mutex_);
    return rel_r_cw_;
  }

  void set_rel_r_cw(const Eigen::Vector4f &r_cw) {
    std::unique_lock lock(mutex_);
    rel_r_cw_ = r_cw;
    updated_at_ = std::chrono::system_clock::now();
  }

  const Eigen::Vector3f get_rel_t_cw() {
    std::shared_lock lock(mutex_);
    return rel_t_cw_;
  }

  void set_rel_t_cw(const Eigen::Vector3f &t_cw) {
    std::unique_lock lock(mutex_);
    rel_t_cw_ = t_cw;
    updated_at_ = std::chrono::system_clock::now();
  }

  const Eigen::Vector4f get_abs_r_cw() {
    std::shared_lock lock(mutex_);
    return abs_r_cw_;
  }

  void set_abs_r_cw(const Eigen::Vector4f &r_cw) {
    std::unique_lock lock(mutex_);
    abs_r_cw_ = r_cw;
    updated_at_ = std::chrono::system_clock::now();
  }

  const Eigen::Vector3f get_abs_t_cw() {
    std::shared_lock lock(mutex_);
    return abs_t_cw_;
  }

  void set_abs_t_cw(const Eigen::Vector3f &t_cw) {
    std::unique_lock lock(mutex_);
    abs_t_cw_ = t_cw;
    updated_at_ = std::chrono::system_clock::now();
  }

  const std::unordered_map<uuids::uuid, Eigen::Vector3f> get_measurement_mps() {
    std::shared_lock lock(mutex_);
    return measurement_mps_;
  }

  void set_measurement_mps(
      const std::unordered_map<uuids::uuid, Eigen::Vector3f> &mps) {
    std::unique_lock lock(mutex_);
    measurement_mps_ = mps;
    updated_at_ = std::chrono::system_clock::now();
  }

  const Ptr get_pre_kf() {
    std::shared_lock lock(mutex_);
    return pre_kf_;
  }

  void set_pre_kf(const Ptr &pre_kf) {
    std::unique_lock lock(mutex_);
    pre_kf_ = pre_kf;
    updated_at_ = std::chrono::system_clock::now();
  }

  const Ptr get_next_kf() {
    std::shared_lock lock(mutex_);
    return next_kf_;
  }

  void set_next_kf(const Ptr &next_kf) {
    std::unique_lock lock(mutex_);
    next_kf_ = next_kf;
    updated_at_ = std::chrono::system_clock::now();
  }

  const std::chrono::system_clock::time_point get_updated_at() {
    std::shared_lock lock(mutex_);
    return updated_at_;
  }

  void set_updated_at(const std::chrono::system_clock::time_point &updated_at) {
    std::unique_lock lock(mutex_);
    updated_at_ = updated_at;
  }

  const std::chrono::system_clock::time_point get_created_at() {
    std::shared_lock lock(mutex_);
    return created_at_;
  }

  void set_created_at(const std::chrono::system_clock::time_point &created_at) {
    std::unique_lock lock(mutex_);
    created_at_ = created_at;
  }

 private:
  uuids::uuid uuid_;
  Eigen::Vector4f rel_r_cw_;
  Eigen::Vector3f rel_t_cw_;
  Eigen::Vector4f abs_r_cw_;
  Eigen::Vector3f abs_t_cw_;

  std::unordered_map<uuids::uuid, Eigen::Vector3f> measurement_mps_;

  Ptr pre_kf_ = nullptr;
  Ptr next_kf_ = nullptr;
  std::chrono::system_clock::time_point updated_at_ =
      std::chrono::system_clock::now();
  std::chrono::system_clock::time_point created_at_ =
      std::chrono::system_clock::now();
  std::shared_mutex mutex_;
};

// template <typename Index,typename Derived>
struct KDAdaptor {
  using Index = std::vector<uuids::uuid>;
  using Derived = std::unordered_map<uuids::uuid, std::shared_ptr<NeoMapPoint>>;
  using coord_t = float;
  //        using point_t = typename Derived::value_type;

  const Index &index_;
  const Derived &obj;  //!< A const ref to the data set origin

  /// The constructor that sets the data set source
  KDAdaptor(const Index &index_, const Derived &obj_)
      : obj(obj_), index_(index_) {}

  /// CRTP helper method
  inline const Derived &derived() const { return obj; }

  inline const Index &index() const { return index_; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return derived().size(); }

  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    auto uuid = index_[idx];
    if (index_[idx] != uuid) {
      return std::numeric_limits<coord_t>::max();
    }
    if (derived().find(uuid) == derived().end() ||
        derived().at(uuid) == nullptr) {
      return std::numeric_limits<coord_t>::max();
    }
    assert(dim < 3);

    return derived().at(uuid)->get_pos()[dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX & /*bb*/) const {
    return false;
  }
};

class NeoMap {
  using MPKDADT = KDAdaptor;
  using MPKDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
      nanoflann::L2_Simple_Adaptor<float, MPKDADT>, MPKDADT, 3>;

 public:
  using Ptr = std::shared_ptr<NeoMap>;
  class DataExistsException : public std::runtime_error {
   public:
    explicit DataExistsException(const uuids::uuid &id)
        : std::runtime_error("UUID " + uuids::to_string(id) +
                             " already exists") {}
  };

  class DataNotFoundException : public std::runtime_error {
   public:
    explicit DataNotFoundException(const uuids::uuid &id)
        : std::runtime_error("UUID " + uuids::to_string(id) + " not found") {}
  };

 public:
  NeoMap() = default;

  std::vector<uuids::uuid> GetMPSUUID() {
    std::shared_lock lock(mutex_);
    // map key to vector
    std::vector<uuids::uuid> ret;
    ret.reserve(mps_.size());
    for (auto &kv : mps_) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  std::vector<uuids::uuid> GetKFSUUID() {
    std::shared_lock lock(mutex_);
    // map key to vector
    std::vector<uuids::uuid> ret;
    ret.reserve(kfs_.size());
    for (auto &kv : kfs_) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  NeoMapPoint::Ptr GetMPByUUID(const uuids::uuid &uuid) {
    if (mps_.find(uuid) == mps_.end()) {
      throw DataNotFoundException(uuid);
    }
    return mps_.at(uuid);
  }

  NeoKeyFrame::Ptr GetKFByUUID(const uuids::uuid &uuid) {
    if (kfs_.find(uuid) == kfs_.end()) {
      throw DataNotFoundException(uuid);
    }
    return kfs_.at(uuid);
  }

 public:
  size_t size() {
    std::shared_lock lock(mutex_);
    return mps_.size();
  }

  void AddMapPoint(const uuids::uuid &uuid, const Eigen::VectorXf &descriptor,
                   const Eigen::Vector3f &pos,
                   const std::pair<float, float> &scale,
                   const std::unordered_set<uuids::uuid> &observations,
                   bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (mps_.find(uuid) != mps_.end()) {
      throw DataExistsException(uuid);
    }
    auto mp = std::make_shared<NeoMapPoint>(uuid, descriptor, pos, scale,
                                            observations);
    mps_[uuid] = mp;
    mps_uuid_vec_.push_back(uuid);
    pos_index_.addPoints(mps_uuid_vec_.size() - 1, mps_uuid_vec_.size() - 1);
    desc_index_.add_feature(uuid, descriptor);
    if (as_neo) {
      neo_mps_queue_.push(uuid);
    }

    lru_cache_.put(uuid, nullptr);
  }

  void UpdateMapPoint(
      const uuids::uuid &uuid,
      const std::optional<Eigen::VectorXf> &descriptor = std::nullopt,
      const std::optional<Eigen::Vector3f> &pos = std::nullopt,
      const std::optional<std::pair<float, float>> &scale = std::nullopt,
      const std::optional<std::unordered_set<uuids::uuid>> &observations =
          std::nullopt,
      bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (mps_.find(uuid) == mps_.end()) {
      throw DataNotFoundException(uuid);
    }

    auto mp = mps_[uuid];
    if (descriptor && mp->get_descriptor() != *descriptor) {
      mp->set_descriptor(*descriptor);
      desc_index_.update_feature(uuid, *descriptor);
    }

    if (pos) {
      mp->set_pos(*pos);
    }
    if (scale) {
      mp->set_scale(*scale);
    }
    if (observations) {
      mp->set_observations(*observations);
    }
    if (as_neo) {
      neo_mps_queue_.push(uuid);
    }
    lru_cache_.put(uuid, nullptr);
  }

  void RemoveMapPoint(const uuids::uuid &uuid) {
    std::unique_lock lock(mutex_);
    if (mps_.find(uuid) == mps_.end()) {
      throw DataNotFoundException(uuid);
    }
    auto mp = mps_[uuid];
    mps_.erase(uuid);
  }

  void AddKeyFrame(const uuids::uuid &uuid, const Eigen::Vector4f &rel_r_cw,
                   const Eigen::Vector3f &rel_t_cw,
                   const Eigen::Vector4f &abs_r_cw,
                   const Eigen::Vector3f &abs_t_cw,
                   const std::unordered_map<uuids::uuid, Eigen::Vector3f> &mps,
                   const NeoKeyFrame::Ptr pre_kf, const bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (kfs_.find(uuid) != kfs_.end()) {
      throw DataExistsException(uuid);
    }
    auto kf = std::make_shared<NeoKeyFrame>(uuid, rel_r_cw, rel_t_cw, abs_r_cw,
                                            abs_t_cw, mps, pre_kf);
    kfs_[uuid] = kf;

    if (as_neo) {
      neo_kfs_queue_.push(uuid);
    }
  }

  void UpdateKeyFrame(
      const uuids::uuid &uuid,
      const std::optional<Eigen::Vector4f> &rel_r_cw = std::nullopt,
      const std::optional<Eigen::Vector3f> &rel_t_cw = std::nullopt,
      const std::optional<Eigen::Vector4f> &abs_r_cw = std::nullopt,
      const std::optional<Eigen::Vector3f> &abs_t_cw = std::nullopt,
      const std::optional<std::unordered_map<uuids::uuid, Eigen::Vector3f>>
          &mps = std::nullopt,
      const bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (kfs_.find(uuid) == kfs_.end()) {
      throw DataNotFoundException(uuid);
    }
    auto kf = kfs_[uuid];
    if (rel_r_cw) {
      kf->set_rel_r_cw(*rel_r_cw);
    }
    if (rel_t_cw) {
      kf->set_rel_t_cw(*rel_t_cw);
    }
    if (abs_r_cw) {
      kf->set_abs_r_cw(*abs_r_cw);
    }
    if (abs_t_cw) {
      kf->set_abs_t_cw(*abs_t_cw);
    }
    if (mps) {
      kf->set_measurement_mps(*mps);
    }
    if (as_neo) {
      neo_kfs_queue_.push(uuid);
    }
  }

  void RemoveKeyFrame(const uuids::uuid &uuid) {
    std::unique_lock lock(mutex_);
    if (kfs_.find(uuid) == kfs_.end()) {
      throw DataNotFoundException(uuid);
    }
    auto kf = kfs_[uuid];
    kfs_.erase(uuid);
  }

  std::vector<uuids::uuid> GetNeighbors(const Eigen::Vector3f &pos,
                                        float radius) {
    if (radius < 0) {
      throw std::invalid_argument("Radius must be non-negative");
    }
    std::shared_lock lock(mutex_);
    float query_pt[3] = {pos[0], pos[1], pos[2]};

    std::vector<nanoflann::ResultItem<size_t, float>> indices_dists;
    nanoflann::RadiusResultSet<float, size_t> result_set(radius, indices_dists);

    pos_index_.findNeighbors(result_set, query_pt);

    std::vector<uuids::uuid> ret;
    ret.reserve(indices_dists.size());
    for (const auto &item : indices_dists) {
      if (mps_.find(mps_uuid_vec_[item.first]) == mps_.end()) {
        continue;
      }
      ret.push_back(mps_uuid_vec_[item.first]);
    }
    return ret;
  }

  std::vector<uuids::uuid> GetNeighbors(uuids::uuid query, float radius) {
    if (radius < 0) {
      throw std::invalid_argument("Radius must be non-negative");
    }

    auto pos = GetMPByUUID(query)->get_pos();

    std::shared_lock lock(mutex_);
    float query_pt[3] = {pos[0], pos[1], pos[2]};

    std::vector<nanoflann::ResultItem<size_t, float>> indices_dists;
    nanoflann::RadiusResultSet<float, size_t> result_set(radius, indices_dists);

    pos_index_.findNeighbors(result_set, query_pt);

    std::vector<uuids::uuid> ret;
    ret.reserve(indices_dists.size());
    for (const auto &item : indices_dists) {
      if (mps_.find(mps_uuid_vec_[item.first]) == mps_.end()) {
        continue;
      }
      if (mps_uuid_vec_[item.first] == query) {
        continue;
      }
      ret.push_back(mps_uuid_vec_[item.first]);
    }
    return ret;
  }

  std::vector<uuids::uuid> GetSimilar(const Eigen::VectorXf &descriptor,
                                      float radius) {
    if (radius < 0) {
      throw std::invalid_argument("Radius must be non-negative");
    }

    std::shared_lock lock(mutex_);
    auto indices = desc_index_.search_range(descriptor, radius);
    std::vector<uuids::uuid> ret;
    ret.reserve(indices.size());
    for (const auto &item : indices) {
      if (mps_.find(item.first) == mps_.end()) {
        continue;
      }
      ret.push_back(item.first);
    }
    return ret;
  }

  std::vector<uuids::uuid> GetHotMPs() {
    std::shared_lock lock(mutex_);
    std::vector<uuids::uuid> ret;
    for (auto [uuid, _] : lru_cache_.get_cache()) {
      ret.push_back(uuid);
    }
    return ret;
  }

 private:
  std::unordered_map<uuids::uuid, NeoMapPoint::Ptr> mps_;
  std::unordered_map<uuids::uuid, NeoKeyFrame::Ptr> kfs_;
  uuids::uuid head_kf_uuid_;

  // 用于kd树的数据
  std::vector<uuids::uuid> mps_uuid_vec_{};
  // 用于多机更新全局位姿
  std::queue<uuids::uuid> neo_mps_queue_;
  std::queue<uuids::uuid> neo_kfs_queue_;

  KDAdaptor kd_adaptor_{mps_uuid_vec_, mps_};
  MPKDTree pos_index_{3, kd_adaptor_};
  FeatureDB desc_index_{32};

  std::shared_mutex mutex_;

  // 用于相对定位检索
  LRUCache<uuids::uuid, void *> lru_cache_{32};
};

// 此处target 希望const怎么办
// TODO 优化效率，大量不需要更新的，不过这个之后通过增量解决吧
static void merge_map(NeoMap::Ptr base, NeoMap::Ptr target) {
  // Compare and update MapPoints
  auto base_mps = base->GetMPSUUID();
  auto target_mps = target->GetMPSUUID();

  std::set<uuids::uuid> base_mps_set(base_mps.begin(), base_mps.end());
  std::set<uuids::uuid> target_mps_set(target_mps.begin(), target_mps.end());

  // add: target - base std::set_difference
  std::set<uuids::uuid> new_mps;
  std::set_difference(target_mps_set.begin(), target_mps_set.end(),
                      base_mps_set.begin(), base_mps_set.end(),
                      std::inserter(new_mps, new_mps.end()));
  for (const auto &uuid : new_mps) {
    try {
      auto mp = target->GetMPByUUID(uuid);
      base->AddMapPoint(uuid, mp->get_descriptor(), mp->get_pos(),
                        mp->get_scale(), mp->get_observations(), true);
    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }
  // remove: base - target
  std::set<uuids::uuid> removed_mps;
  std::set_difference(base_mps_set.begin(), base_mps_set.end(),
                      target_mps_set.begin(), target_mps_set.end(),
                      std::inserter(removed_mps, removed_mps.end()));
  for (const auto &uuid : removed_mps) {
    try {
      base->RemoveMapPoint(uuid);

    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  std::set<uuids::uuid> updated_mps;
  std::set_intersection(base_mps_set.begin(), base_mps_set.end(),
                        target_mps_set.begin(), target_mps_set.end(),
                        std::inserter(updated_mps, updated_mps.end()));
  for (const auto &uuid : updated_mps) {
    try {
      auto mp = target->GetMPByUUID(uuid);
      auto base_mp = base->GetMPByUUID(uuid);
      if (mp->get_descriptor() == base_mp->get_descriptor() &&
          mp->get_scale() == base_mp->get_scale() &&
          mp->get_observations() == base_mp->get_observations()) {
        continue;
      }
      base->UpdateMapPoint(uuid, mp->get_descriptor(), std::nullopt,
                           mp->get_scale(), mp->get_observations(), true);
    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  auto base_kfs = base->GetKFSUUID();
  auto target_kfs = target->GetKFSUUID();

  std::set<uuids::uuid> base_kfs_set(base_kfs.begin(), base_kfs.end());
  std::set<uuids::uuid> target_kfs_set(target_kfs.begin(), target_kfs.end());

  std::set<uuids::uuid> new_kfs;
  // add: target - base std::set_difference
  std::set_difference(target_kfs_set.begin(), target_kfs_set.end(),
                      base_kfs_set.begin(), base_kfs_set.end(),
                      std::inserter(new_kfs, new_kfs.end()));

  for (const auto &uuid : new_kfs) {
    try {
      auto kf = target->GetKFByUUID(uuid);
      base->AddKeyFrame(uuid, kf->get_rel_r_cw(), kf->get_rel_t_cw(),
                        kf->get_abs_r_cw(), kf->get_abs_t_cw(),
                        kf->get_measurement_mps(), kf->get_pre_kf(), true);
    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }
  std::set<uuids::uuid> removed_kfs;
  std::set_difference(base_kfs_set.begin(), base_kfs_set.end(),
                      target_kfs_set.begin(), target_kfs_set.end(),
                      std::inserter(removed_kfs, removed_kfs.end()));
  for (const auto &uuid : removed_kfs) {
    try {
      base->RemoveKeyFrame(uuid);
    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  std::set<uuids::uuid> updated_kfs;
  std::set_intersection(base_kfs_set.begin(), base_kfs_set.end(),
                        target_kfs_set.begin(), target_kfs_set.end(),
                        std::inserter(updated_kfs, updated_kfs.end()));

  for (const auto &uuid : updated_kfs) {
    try {
      auto kf = target->GetKFByUUID(uuid);
      auto base_kf = base->GetKFByUUID(uuid);
      if (kf->get_rel_r_cw() == base_kf->get_rel_r_cw() &&
          kf->get_rel_t_cw() == base_kf->get_rel_t_cw() &&
          kf->get_measurement_mps() == base_kf->get_measurement_mps()) {
        continue;
      }
      base->UpdateKeyFrame(uuid, kf->get_rel_r_cw(), kf->get_rel_t_cw(),
                           std::nullopt, std::nullopt,
                           kf->get_measurement_mps(), true

      );
    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }
}

}  // namespace rgor

#endif  // RGOR_NEO_MAP_H
