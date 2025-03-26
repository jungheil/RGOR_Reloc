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
#include <fstream>
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
#include "proto/PMap.pb.h"
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
              std::set<uuids::uuid> observations,
              std::chrono::system_clock::time_point updated_at =
                  std::chrono::system_clock::now(),
              std::chrono::system_clock::time_point created_at =
                  std::chrono::system_clock::now())
      : uuid_(uuid),
        descriptor_(descriptor),
        pos_(pos),
        scale_(scale),
        observations_(observations),
        updated_at_(updated_at),
        created_at_(created_at) {}

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

  const Eigen::Vector3f get_gb_pos() {
    std::shared_lock lock(mutex_);
    return gb_pos_;
  }

  void set_gb_pos(const Eigen::Vector3f &pos) {
    std::unique_lock lock(mutex_);
    gb_pos_ = pos;
  }

  void set_gb_init(bool gb_init) {
    std::unique_lock lock(mutex_);
    gb_init_ = gb_init;
  }

  const bool get_gb_init() {
    std::shared_lock lock(mutex_);
    return gb_init_;
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
  const std::set<uuids::uuid> get_observations() {
    std::shared_lock lock(mutex_);
    return observations_;
  }

  void set_observations(const std::set<uuids::uuid> &observations) {
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
  alignas(64) std::pair<float, float> scale_;
  std::set<uuids::uuid> observations_;
  Eigen::Vector3f pos_;
  bool gb_init_ = false;
  Eigen::Vector3f gb_pos_;

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
              uuids::uuid pre_kf = uuids::uuid(),
              uuids::uuid next_kf = uuids::uuid(),
              std::chrono::system_clock::time_point updated_at =
                  std::chrono::system_clock::now(),
              std::chrono::system_clock::time_point created_at =
                  std::chrono::system_clock::now())
      : uuid_(uuid),
        rel_r_cw_(rel_r_cw),
        rel_t_cw_(rel_t_cw),
        abs_r_cw_(abs_r_cw),
        abs_t_cw_(abs_t_cw),
        measurement_mps_(mps),
        pre_kf_(pre_kf),
        next_kf_(next_kf),
        updated_at_(updated_at),
        created_at_(created_at) {}

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

  const Eigen::Vector3f get_gb_t_cw() {
    std::shared_lock lock(mutex_);
    return gb_t_cw_;
  }

  void set_gb_t_cw(const Eigen::Vector3f &t_cw) {
    std::unique_lock lock(mutex_);
    gb_t_cw_ = t_cw;
  }

  const Eigen::Vector4f get_gb_r_cw() {
    std::shared_lock lock(mutex_);
    return gb_r_cw_;
  }

  void set_gb_r_cw(const Eigen::Vector4f &r_cw) {
    std::unique_lock lock(mutex_);
    gb_r_cw_ = r_cw;
  }

  void set_gb_init(bool gb_init) {
    std::unique_lock lock(mutex_);
    gb_init_ = gb_init;
  }

  const bool get_gb_init() {
    std::shared_lock lock(mutex_);
    return gb_init_;
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

  const uuids::uuid get_pre_kf() {
    std::shared_lock lock(mutex_);
    return pre_kf_;
  }

  void set_pre_kf(const uuids::uuid pre_kf) {
    std::unique_lock lock(mutex_);
    pre_kf_ = pre_kf;
    updated_at_ = std::chrono::system_clock::now();
  }

  const uuids::uuid get_next_kf() {
    std::shared_lock lock(mutex_);
    return next_kf_;
  }

  void set_next_kf(const uuids::uuid next_kf) {
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

  bool gb_init_ = false;
  Eigen::Vector4f gb_r_cw_;
  Eigen::Vector3f gb_t_cw_;

  std::unordered_map<uuids::uuid, Eigen::Vector3f> measurement_mps_;

  uuids::uuid pre_kf_;
  uuids::uuid next_kf_;
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
  explicit NeoMap() {
    // 生成UUID
    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 engine(seq);
    uuids::uuid_random_generator gen(&engine);
    uuid_ = gen();
  }

  explicit NeoMap(uuids::uuid uuid,
                  std::chrono::system_clock::time_point created_at =
                      std::chrono::system_clock::now())
      : uuid_(uuid) , created_at_(created_at) {}

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

  const uuids::uuid get_uuid() {
    std::shared_lock lock(mutex_);
    return uuid_;
  }

  const std::unordered_set<uuids::uuid> get_new_kfs_set() {
    std::shared_lock lock(mutex_);
    return new_kfs_set_;
  }

  void remove_new_kfs_set(const uuids::uuid &uuid) {
    std::unique_lock lock(mutex_);
    new_kfs_set_.erase(uuid);
  }

  void clear_new_kfs_set() {
    std::unique_lock lock(mutex_);
    new_kfs_set_.clear();
  }

  const std::unordered_set<uuids::uuid> get_updated_kfs_set() {
    std::shared_lock lock(mutex_);
    return updated_kfs_set_;
  }

  void remove_updated_kfs_set(const uuids::uuid &uuid) {
    std::unique_lock lock(mutex_);
    updated_kfs_set_.erase(uuid);
  }

  void clear_updated_kfs_set() {
    std::unique_lock lock(mutex_);
    updated_kfs_set_.clear();
  }

  void insert_updated_kfs_set(const uuids::uuid &uuid) {
    std::unique_lock lock(mutex_);
    updated_kfs_set_.insert(uuid);
  }

  const bool get_gb_init() {
    std::shared_lock lock(mutex_);
    return gb_init_;
  }

  void set_gb_init(bool gb_init) {
    std::unique_lock lock(mutex_);
    gb_init_ = gb_init;
  }

      public : size_t
               size() {
    std::shared_lock lock(mutex_);
    return mps_.size();
  }

  const std::chrono::system_clock::time_point get_created_at() {
    std::shared_lock lock(mutex_);
    return created_at_;
  }

  const uuids::uuid get_gb_uuid() {
    std::shared_lock lock(mutex_);
    return gb_uuid_;
  }

        void set_gb_uuid(const uuids::uuid &uuid) {
        std::unique_lock lock(mutex_);
        gb_uuid_ = uuid;
        }

        // get mps

        const std::unordered_map<uuids::uuid, NeoMapPoint::Ptr> &get_mps() {

                std::shared_lock lock(mutex_);
                return mps_;
        }

        // get kfs
        const std::unordered_map<uuids::uuid, NeoKeyFrame::Ptr> &get_kfs() {

                std::shared_lock lock(mutex_);
                return kfs_;
        }



  void AddMapPoint(const uuids::uuid &uuid, const Eigen::VectorXf &descriptor,
                   const Eigen::Vector3f &pos,
                   const std::pair<float, float> &scale,
                   const std::set<uuids::uuid> &observations,
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
    // Note: 用不上，统一kf更新
    if (as_neo) {
      new_mps_set_.insert(uuid);
    }

    lru_cache_.put(uuid, nullptr);
    new_mp_count_++;
  }

  bool UpdateMapPoint(
      const uuids::uuid &uuid,
      const std::optional<Eigen::VectorXf> &descriptor = std::nullopt,
      const std::optional<Eigen::Vector3f> &pos = std::nullopt,
      const std::optional<std::pair<float, float>> &scale = std::nullopt,
      const std::optional<std::set<uuids::uuid>> &observations =
          std::nullopt,
      bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (mps_.find(uuid) == mps_.end()) {
      throw DataNotFoundException(uuid);
    }

    bool ret = false;

    auto mp = mps_[uuid];
    if (descriptor && mp->get_descriptor() != *descriptor) {
      mp->set_descriptor(*descriptor);
      desc_index_.update_feature(uuid, *descriptor);
      ret = true;
    }

    if (pos && mp->get_pos() != *pos) {
      mp->set_pos(*pos);
      ret = true;
    }
    if (scale && mp->get_scale() != *scale) {
      mp->set_scale(*scale);
      ret = true;
    }
    if (observations && mp->get_observations() != *observations) {
      mp->set_observations(*observations);
      ret = true;
    }
    // if (as_neo && ret) {

    // }
    lru_cache_.put(uuid, nullptr);
    new_mp_count_++;
    return ret;
  }

  void RemoveMapPoint(const uuids::uuid &uuid) {
    std::unique_lock lock(mutex_);
    if (mps_.find(uuid) == mps_.end()) {
      throw DataNotFoundException(uuid);
    }
    auto mp = mps_[uuid];
    mps_.erase(uuid);
  }

  void AddKeyFrame(
      const uuids::uuid &uuid, const Eigen::Vector4f &rel_r_cw,
      const Eigen::Vector3f &rel_t_cw, const Eigen::Vector4f &abs_r_cw,
      const Eigen::Vector3f &abs_t_cw,
      const std::unordered_map<uuids::uuid, Eigen::Vector3f> &mps,
      const uuids::uuid pre_kf = uuids::uuid(),
      const uuids::uuid next_kf = uuids::uuid(),
      const std::chrono::system_clock::time_point updated_at =
          std::chrono::system_clock::now(),
      const std::chrono::system_clock::time_point created_at =
          std::chrono::system_clock::now(),
      const bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (kfs_.find(uuid) != kfs_.end()) {
      throw DataExistsException(uuid);
    }
    auto kf = std::make_shared<NeoKeyFrame>(uuid, rel_r_cw, rel_t_cw, abs_r_cw,
                                            abs_t_cw, mps, pre_kf);
    kfs_[uuid] = kf;

    if (as_neo) {
      new_kfs_set_.insert(uuid);
    }
  }

  bool UpdateKeyFrame(
      const uuids::uuid &uuid,
      const std::optional<Eigen::Vector4f> &rel_r_cw = std::nullopt,
      const std::optional<Eigen::Vector3f> &rel_t_cw = std::nullopt,
      const std::optional<Eigen::Vector4f> &abs_r_cw = std::nullopt,
      const std::optional<Eigen::Vector3f> &abs_t_cw = std::nullopt,
      const std::optional<std::unordered_map<uuids::uuid, Eigen::Vector3f>>
          &mps = std::nullopt,
      const std::optional<uuids::uuid> &pre_kf = std::nullopt,
      const std::optional<uuids::uuid> &next_kf = std::nullopt,
      const bool as_neo = false) {
    std::unique_lock lock(mutex_);
    if (kfs_.find(uuid) == kfs_.end()) {
      throw DataNotFoundException(uuid);
    }
    bool ret = false;
    auto kf = kfs_[uuid];
    if (rel_r_cw && kf->get_rel_r_cw() != *rel_r_cw) {
      kf->set_rel_r_cw(*rel_r_cw);
      ret = true;
    }
    if (rel_t_cw && kf->get_rel_t_cw() != *rel_t_cw) {
      kf->set_rel_t_cw(*rel_t_cw);
      ret = true;
    }
    if (abs_r_cw && kf->get_abs_r_cw() != *abs_r_cw) {
      kf->set_abs_r_cw(*abs_r_cw);
      ret = true;
    }
    if (abs_t_cw && kf->get_abs_t_cw() != *abs_t_cw) {
      kf->set_abs_t_cw(*abs_t_cw);
      ret = true;
    }
    if (mps && kf->get_measurement_mps() != *mps) {
      kf->set_measurement_mps(*mps);
      ret = true;
    }
    if (pre_kf && kf->get_pre_kf() != *pre_kf) {
      kf->set_pre_kf(*pre_kf);
      ret = true;
    }
    if (next_kf && kf->get_next_kf() != *next_kf) {
      kf->set_next_kf(*next_kf);
      ret = true;
    }
    if (as_neo && ret) {
      updated_kfs_set_.insert(uuid);
    }
    return ret;
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

  size_t get_new_mp_count() {
    std::shared_lock lock(mutex_);
    return new_mp_count_;
  }

  void ClearHotMPs(){
    lru_cache_.clear();
    new_mp_count_ = 0;
  }

  void Dump(std::string_view path) {
    std::shared_lock lock(mutex_);

    // 创建protobuf消息
    ::NeoMap proto_map;

    // 添加所有MapPoints
    for (const auto &[uuid, mp] : mps_) {
      auto *proto_mp = proto_map.add_mps();

      // 设置UUID
      proto_mp->set_uuid(uuid.as_bytes().data(), 16);

      // 设置描述子
      const auto desc = mp->get_descriptor();
      proto_mp->set_desc(desc.data(), desc.size() * sizeof(float));

      // 设置位置
      auto *proto_pose = proto_mp->mutable_pose();
      const auto &pos = mp->get_pos();
      proto_pose->set_x(pos[0]);
      proto_pose->set_y(pos[1]);
      proto_pose->set_z(pos[2]);

      // 设置全局初始化标志和全局位姿
      proto_mp->set_gb_init(mp->get_gb_init());
      if(mp->get_gb_init()) {
        auto *proto_gb_pose = proto_mp->mutable_gb_pose();
        const auto &gb_pos = mp->get_gb_pos();
        proto_gb_pose->set_x(gb_pos[0]);
        proto_gb_pose->set_y(gb_pos[1]);
        proto_gb_pose->set_z(gb_pos[2]);
      }

      // 设置尺度
      auto *proto_scale = proto_mp->mutable_scale();
      const auto &scale = mp->get_scale();
      proto_scale->set_s(scale.first);
      proto_scale->set_l(scale.second);

      // 设置观测帧
      for (const auto &obs_uuid : mp->get_observations()) {
        proto_mp->add_observations(obs_uuid.as_bytes().data(), 16);
      }

      // 设置时间戳
      proto_mp->set_updated_at(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              mp->get_updated_at().time_since_epoch())
              .count());
      proto_mp->set_created_at(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              mp->get_created_at().time_since_epoch())
              .count());
    }

    // 添加所有KeyFrames
    for (const auto &[uuid, kf] : kfs_) {
      auto *proto_kf = proto_map.add_kfs();

      // 设置UUID
      proto_kf->set_uuid(uuid.as_bytes().data(), 16);

      // 设置相对位姿
      auto *proto_pose_rel = proto_kf->mutable_pose_rel();
      const auto &rel_t = kf->get_rel_t_cw();
      proto_pose_rel->set_x(rel_t[0]);
      proto_pose_rel->set_y(rel_t[1]);
      proto_pose_rel->set_z(rel_t[2]);

      auto *proto_rot_rel = proto_kf->mutable_rotation_rel();
      const auto &rel_r = kf->get_rel_r_cw();
      proto_rot_rel->set_w(rel_r[0]);
      proto_rot_rel->set_x(rel_r[1]);
      proto_rot_rel->set_y(rel_r[2]);
      proto_rot_rel->set_z(rel_r[3]);

      // 设置绝对位姿
      auto *proto_pose_abs = proto_kf->mutable_pose_abs();
      const auto &abs_t = kf->get_abs_t_cw();
      proto_pose_abs->set_x(abs_t[0]);
      proto_pose_abs->set_y(abs_t[1]);
      proto_pose_abs->set_z(abs_t[2]);

      auto *proto_rot_abs = proto_kf->mutable_rotation_abs();
      const auto &abs_r = kf->get_abs_r_cw();
      proto_rot_abs->set_w(abs_r[0]);
      proto_rot_abs->set_x(abs_r[1]);
      proto_rot_abs->set_y(abs_r[2]);
      proto_rot_abs->set_z(abs_r[3]);

      // 设置全局初始化标志和全局位姿
      proto_kf->set_gb_init(kf->get_gb_init());
      if(kf->get_gb_init()) {
        auto *proto_gb_pose = proto_kf->mutable_gb_pose();
        const auto &gb_t = kf->get_gb_t_cw();
        proto_gb_pose->set_x(gb_t[0]);
        proto_gb_pose->set_y(gb_t[1]);
        proto_gb_pose->set_z(gb_t[2]);

        auto *proto_gb_rot = proto_kf->mutable_gb_rotation();
        const auto &gb_r = kf->get_gb_r_cw();
        proto_gb_rot->set_w(gb_r[0]);
        proto_gb_rot->set_x(gb_r[1]);
        proto_gb_rot->set_y(gb_r[2]);
        proto_gb_rot->set_z(gb_r[3]);
      }

      // 设置相邻帧
      proto_kf->set_pre_kf(kf->get_pre_kf().as_bytes().data(), 16);
      proto_kf->set_next_kf(kf->get_next_kf().as_bytes().data(), 16);

      // 设置观测到的MapPoints
      for (const auto &[mp_uuid, mp_pos] : kf->get_measurement_mps()) {
        auto *measurement = proto_kf->add_measurement();
        measurement->set_uuid(mp_uuid.as_bytes().data(), 16);
        auto *mp_proto_pose = measurement->mutable_pose();
        mp_proto_pose->set_x(mp_pos[0]);
        mp_proto_pose->set_y(mp_pos[1]);
        mp_proto_pose->set_z(mp_pos[2]);
      }

      // 设置时间戳
      proto_kf->set_updated_at(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              kf->get_updated_at().time_since_epoch())
              .count());
      proto_kf->set_created_at(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              kf->get_created_at().time_since_epoch())
              .count());
    }

    // 设置全局标记
    proto_map.set_gb_init(gb_init_);
    if(gb_init_) {
      proto_map.set_gb_uuid(gb_uuid_.as_bytes().data(), 16);
    }

    // 序列化并保存到文件
    std::ofstream fout(path.data(), std::ios::binary);
    if (!fout.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(path));
    }
    if (!proto_map.SerializeToOstream(&fout)) {
      throw std::runtime_error("Failed to write to file " + std::string(path));
    }
  }

  void Load(std::string_view path) {
    std::unique_lock lock(mutex_);

    // 从文件读取并解析protobuf消息
    ::NeoMap proto_map;
    std::ifstream fin(path.data(), std::ios::binary);
    if (!fin.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(path));
    }
    if (!proto_map.ParseFromIstream(&fin)) {
      throw std::runtime_error("Failed to parse file " + std::string(path));
    }

    // 清空当前数据
    mps_.clear();
    kfs_.clear();
    mps_uuid_vec_.clear();

    // 加载MapPoints
    std::array<unsigned char, 16> uuid_bytes;

    for (const auto &proto_mp : proto_map.mps()) {
      // 解析UUID
      for (int i = 0; i < 16; i++) {
        uuid_bytes[i] = proto_mp.uuid().data()[i];
      }
      uuids::uuid uuid = uuids::uuid(uuid_bytes);

      Eigen::VectorXf desc(proto_mp.desc().size() / sizeof(float));
      std::memcpy(desc.data(), proto_mp.desc().data(), proto_mp.desc().size());

      // 解析位置
      Eigen::Vector3f pos;
      pos << proto_mp.pose().x(), proto_mp.pose().y(), proto_mp.pose().z();

      // 解析尺度
      std::pair<float, float> scale{proto_mp.scale().s(), proto_mp.scale().l()};

      // 解析观测帧
      std::set<uuids::uuid> observations;
      for (const auto &obs_bytes : proto_mp.observations()) {
        for (int i = 0; i < 16; i++) {
          uuid_bytes[i] = proto_mp.uuid().data()[i];
        }
        observations.insert(uuids::uuid(uuid_bytes));
      }

      // 添加MapPoint
      AddMapPoint(uuid, desc, pos, scale, observations);

      // 设置时间戳
      auto mp = mps_[uuid];
      mp->set_updated_at(std::chrono::system_clock::time_point(
          std::chrono::milliseconds(proto_mp.updated_at())));
      mp->set_created_at(std::chrono::system_clock::time_point(
          std::chrono::milliseconds(proto_mp.created_at())));
          
      // 设置全局初始化标记和全局位姿
      mp->set_gb_init(proto_mp.gb_init());
      if(proto_mp.has_gb_pose()) {
        Eigen::Vector3f gb_pos;
        gb_pos << proto_mp.gb_pose().x(), proto_mp.gb_pose().y(), proto_mp.gb_pose().z();
        mp->set_gb_pos(gb_pos);
      }
    }

    // 加载KeyFrames
    for (const auto &proto_kf : proto_map.kfs()) {
      // 解析UUID
      for (int i = 0; i < 16; i++) {
        uuid_bytes[i] = proto_kf.uuid().data()[i];
      }
      uuids::uuid uuid = uuids::uuid(uuid_bytes);

      // 解析相对位姿
      Eigen::Vector4f rel_r;
      rel_r << proto_kf.rotation_rel().w(), proto_kf.rotation_rel().x(),
          proto_kf.rotation_rel().y(), proto_kf.rotation_rel().z();
      Eigen::Vector3f rel_t;
      rel_t << proto_kf.pose_rel().x(), proto_kf.pose_rel().y(),
          proto_kf.pose_rel().z();

      // 解析绝对位姿
      Eigen::Vector4f abs_r;
      abs_r << proto_kf.rotation_abs().w(), proto_kf.rotation_abs().x(),
          proto_kf.rotation_abs().y(), proto_kf.rotation_abs().z();
      Eigen::Vector3f abs_t;
      abs_t << proto_kf.pose_abs().x(), proto_kf.pose_abs().y(),
          proto_kf.pose_abs().z();

      // 解析观测到的MapPoints
      std::unordered_map<uuids::uuid, Eigen::Vector3f> mps;
      for (const auto &measurement : proto_kf.measurement()) {
        for (int i = 0; i < 16; i++) {
          uuid_bytes[i] = measurement.uuid().data()[i];
        }
        uuids::uuid mp_uuid = uuids::uuid(uuid_bytes);
        Eigen::Vector3f mp_pos;
        mp_pos << measurement.pose().x(), measurement.pose().y(),
            measurement.pose().z();
        mps[mp_uuid] = mp_pos;
      }

      // 解析相邻帧
      for (int i = 0; i < 16; i++) {
        uuid_bytes[i] = proto_kf.pre_kf().data()[i];
      }
      uuids::uuid pre_kf = uuids::uuid(uuid_bytes);

      for (int i = 0; i < 16; i++) {
        uuid_bytes[i] = proto_kf.next_kf().data()[i];
      }
      uuids::uuid next_kf = uuids::uuid(uuid_bytes);

      // 添加KeyFrame
      AddKeyFrame(uuid, rel_r, rel_t, abs_r, abs_t, mps, pre_kf, next_kf);

      // 设置时间戳
      auto kf = kfs_[uuid];
      kf->set_updated_at(std::chrono::system_clock::time_point(
          std::chrono::milliseconds(proto_kf.updated_at())));
      kf->set_created_at(std::chrono::system_clock::time_point(
          std::chrono::milliseconds(proto_kf.created_at())));

      // 设置全局初始化标记和全局位姿
      kf->set_gb_init(proto_kf.gb_init());
      if(proto_kf.has_gb_pose()) {
        Eigen::Vector3f gb_t;
        gb_t << proto_kf.gb_pose().x(), proto_kf.gb_pose().y(), proto_kf.gb_pose().z();
        kf->set_gb_t_cw(gb_t);
      }
      if(proto_kf.has_gb_rotation()) {
        Eigen::Vector4f gb_r;
        gb_r << proto_kf.gb_rotation().w(), proto_kf.gb_rotation().x(),
             proto_kf.gb_rotation().y(), proto_kf.gb_rotation().z();
        kf->set_gb_r_cw(gb_r);
      }
    }

    // 设置Map的全局初始化标记
    gb_init_ = proto_map.gb_init();
    if(proto_map.has_gb_uuid()) {
      for(int i = 0; i < 16; i++) {
        uuid_bytes[i] = proto_map.gb_uuid().data()[i];
      }
      gb_uuid_ = uuids::uuid(uuid_bytes);
    }

    // 重建索引
    for (const auto &[uuid, _] : mps_) {
      mps_uuid_vec_.push_back(uuid);
      pos_index_.addPoints(mps_uuid_vec_.size() - 1, mps_uuid_vec_.size() - 1);
      desc_index_.add_feature(uuid, mps_[uuid]->get_descriptor());
    }
  }

 private:
  uuids::uuid uuid_;
  std::unordered_map<uuids::uuid, NeoMapPoint::Ptr> mps_;
  std::unordered_map<uuids::uuid, NeoKeyFrame::Ptr> kfs_;
  uuids::uuid head_kf_uuid_;

  // 用于kd树的数据
  std::vector<uuids::uuid> mps_uuid_vec_{};
  // 用于多机更新全局位姿
  std::unordered_set<uuids::uuid> new_mps_set_;
  std::unordered_set<uuids::uuid> new_kfs_set_;
  std::unordered_set<uuids::uuid> updated_kfs_set_;

  KDAdaptor kd_adaptor_{mps_uuid_vec_, mps_};
  MPKDTree pos_index_{3, kd_adaptor_};
  FeatureDB desc_index_{32};

  bool gb_init_ = false;
  uuids::uuid gb_uuid_;

  std::shared_mutex mutex_;

  // 用于相对定位检索
  LRUCache<uuids::uuid, void *> lru_cache_{32};
  size_t new_mp_count_ = 0;

  std::chrono::system_clock::time_point created_at_;
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
                        mp->get_scale(), mp->get_observations());
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
      base->UpdateMapPoint(uuid, mp->get_descriptor(), mp->get_pos(),
                           mp->get_scale(), mp->get_observations());
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
                        kf->get_measurement_mps(), kf->get_pre_kf(),
                        uuids::uuid(), kf->get_updated_at(), kf->get_created_at());  
      base->UpdateKeyFrame(kf->get_pre_kf(), std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, std::nullopt, uuid, false);
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
                           kf->get_abs_r_cw(), kf->get_abs_t_cw(),
                           kf->get_measurement_mps(), kf->get_pre_kf(),
                           kf->get_next_kf());
    } catch (NeoMap::DataNotFoundException &e) {
      std::cerr << e.what() << std::endl;
    }
  }
}

}  // namespace rgor

#endif  // RGOR_NEO_MAP_H
