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

#ifndef RGOR_MAP_H
#define RGOR_MAP_H

#include <Eigen/Core>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
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

namespace rgor {
class Camera;

class KeyFrame;

class Frame;

template <typename Derived>
struct PointCloudAdaptor {
  using coord_t = float;
  //        using point_t = typename Derived::value_type;

  const Derived &obj;  //!< A const ref to the data set origin

  /// The constructor that sets the data set source
  PointCloudAdaptor(const Derived &obj_) : obj(obj_) {}

  /// CRTP helper method
  inline const Derived &derived() const { return obj; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return derived().size(); }

  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (derived()[idx] == nullptr) {
      // return 0;
      return std::numeric_limits<coord_t>::max();
    }
    assert(dim < 3);
    return derived()[idx]->get_pos()[dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX & /*bb*/) const {
    return false;
  }
};

/**
 * @brief 地图点类模板
 * @tparam frame_type 关键帧类型,默认为KeyFrame
 */
template <typename frame_type = KeyFrame>
class alignas(64) MapPoint {
 public:
  using Ptr = std::shared_ptr<MapPoint<frame_type>>;
  using coord_t = float;
  static std::mutex id_counter_mutex_;  // 用于保护id计数器的互斥锁

  /**
   * @brief 构造函数
   * @param pos 3D位置
   * @param scale 特征点尺度
   * @param descriptor 描述子
   * @param fixed 是否固定(不优化)
   */
  MapPoint(Eigen::Vector3f pos, std::pair<float, float> scale,
           Eigen::VectorXf descriptor, bool fixed = false)
      : pos_(std::move(pos)),
        scale_(scale),
        descriptor_(descriptor),
        fixed_(fixed) {
    {
      std::lock_guard<std::mutex> lck(id_counter_mutex_);
      id_ = ++id_counter_;  // 原子操作生成唯一ID
    }
    // 生成UUID
    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 engine(seq);
    uuids::uuid_random_generator gen(&engine);
    uuid_ = gen();
    cls_ = GetTopCls(descriptor_);  // 计算描述子的top-k类别
  }

  // 禁用拷贝构造函数
  MapPoint(const MapPoint &mp) = delete;

  // Getter/Setter方法
  size_t get_id() const { return id_; }

  uuids::uuid get_uuid() const { return uuid_; }

  /**
   * @brief 获取3D位置
   * @return 3D位置向量
   */
  Eigen::Vector3f get_pos() const {
    std::shared_lock<std::shared_mutex> lock(data_mutex_);
    return pos_;
  }

  /**
   * @brief 设置3D位置
   * @param pos 新的3D位置
   */
  void set_pos(const Eigen::Vector3f &pos) {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);
    pos_ = pos;
  }

  /**
   * @brief 获取特征点尺度
   * @return 尺度对(min,max)
   */
  std::pair<float, float> get_scale() {
    std::shared_lock<std::shared_mutex> lock(data_mutex_);
    return scale_;
  }

  /**
   * @brief 设置特征点尺度
   * @param scale 新的尺度对
   */
  void set_scale(std::pair<float, float> scale) {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);
    scale_ = scale;
  }

  /**
   * @brief 获取描述子
   * @return 描述子向量
   */
  const Eigen::VectorXf get_descriptor() {
    std::shared_lock<std::shared_mutex> lock(data_mutex_);
    // TODO确认是否竞争
    return descriptor_;
  }

  /**
   * @brief 设置描述子
   * @param descriptor 新的描述子
   */
  void SetDescriptor(Eigen::VectorXf descriptor) {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);
    if (descriptor_.size()) {
      if (descriptor.size() != descriptor_.size()) {
        throw std::invalid_argument("Invalid descriptor size");
      }
      descriptor_ = descriptor;

    } else {
      descriptor_ = descriptor;
    }
    cls_ = GetTopCls(descriptor_);
  }

  /**
   * @brief 获取描述子的top-k类别
   * @return 类别数组
   */
  const std::array<size_t, MAPPOINT_TOP_K> get_cls() const { return cls_; }

  /**
   * @brief 获取观测到该点的所有关键帧
   * @return 关键帧ID到<关键帧,特征点索引>对的映射
   */
  std::unordered_map<uuids::uuid, std::pair<std::weak_ptr<frame_type>, size_t>>
  get_observations() const {
    std::shared_lock<std::shared_mutex> lock(data_mutex_);
    return observations_;
  }

  /**
   * @brief 添加一个观测
   * @param frame 观测到该点的关键帧
   * @param idx 特征点在关键帧中的索引
   */
  void AddObservation(std::shared_ptr<frame_type> frame, size_t idx) {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);
    if (frame == nullptr) {
      return;
    }
    observations_[frame->get_uuid()] = {frame, idx};
  }

  /**
   * @brief 删除一个观测
   * @param frame 要删除的关键帧
   */
  void EraseObservation(std::shared_ptr<frame_type> frame) {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);
    observations_.erase(frame->get_uuid());
  }

  // 状态设置/获取
  void set_on_map(bool on_map) { on_map_ = on_map; }

  const bool get_on_map() const { return on_map_; }

  void set_bad(bool bad) { bad_ = bad; }

  const bool get_bad() const { return bad_; }

  void set_fixed(bool fixed) { fixed_ = fixed; }

  const bool get_fixed() const { return fixed_; }

  void set_map_id(size_t id) {
    if (id >= std::numeric_limits<size_t>::max()) {
      throw std::out_of_range("Invalid map_id");
    }
    map_id = id;
  }

  const size_t get_map_id() const { return map_id; }

  // 观测计数相关方法
  void IncreaseObservedTimes() {
    ++observed_times_;
    ++recent_observed_times_;
  }

  void IncreaseRealObservedTimes() { ++real_observed_times_; }

  void ClearRecentObservedTimes() { recent_observed_times_ = 0; }

  size_t get_recent_observed_times() { return recent_observed_times_; }

  /**
   * @brief 获取实际观测次数与应该观测次数的比值
   * @return 观测比例
   */
  float GetObservedRatio() {
    return static_cast<float>(real_observed_times_) / observed_times_;
  }

 private:
  /**
   * @brief 计算描述子的top-k类别
   * @param descriptor 输入描述子
   * @return top-k类别数组
   */
  std::array<size_t, MAPPOINT_TOP_K> GetTopCls(Eigen::VectorXf descriptor) {
    std::array<size_t, MAPPOINT_TOP_K> cls;
    cls.fill(0);

    if (descriptor.size() == 0) {
      return cls;
    }

    Eigen::VectorXf desc_copy = descriptor;

    // 找出描述子中最大的k个值对应的索引
    for (size_t i = 0; i < MAPPOINT_TOP_K && i < descriptor.size(); ++i) {
      float max = 0;
      size_t max_idx = 0;
      for (size_t j = 0; j < desc_copy.size(); ++j) {
        if (desc_copy[j] > max) {
          max = desc_copy[j];
          max_idx = j;
        }
      }
      cls[i] = max_idx;
      desc_copy[max_idx] = 0;  // 将已找到的最大值置0
    }
    return cls;
  }

 private:
  size_t id_ = 0;             // 地图点ID
  static size_t id_counter_;  // 全局ID计数器
  uuids::uuid uuid_;          // 唯一标识符

  alignas(64) Eigen::Vector3f pos_;            // 3D位置
  alignas(64) std::pair<float, float> scale_;  // 特征点尺度
  // NOTE 改变descriptor可能会有阈值和数值范围问题
  Eigen::VectorXf descriptor_;              // 描述子
  std::array<size_t, MAPPOINT_TOP_K> cls_;  // 描述子的top-k类别

  // 观测到该点的所有关键帧
  std::unordered_map<uuids::uuid, std::pair<std::weak_ptr<frame_type>, size_t>>
      observations_{};

  bool on_map_ = false;  // 是否在地图中
  bool bad_ = false;     // 是否是坏点
  bool fixed_ = false;   // 是否固定(不优化)
  size_t map_id = -1;    // 在地图数组中的索引

  // 观测统计
  size_t observed_times_ = 1;         // 应该被观测的次数
  size_t recent_observed_times_ = 1;  // 最近被观测的次数
  size_t real_observed_times_ = 1;    // 实际被观测的次数

  mutable std::shared_mutex data_mutex_;  // 保护数据成员的读写锁
};

// 静态成员初始化
template <typename T>
size_t MapPoint<T>::id_counter_ = 0;

template <typename T>
std::mutex MapPoint<T>::id_counter_mutex_;

class Map {
 public:
  using Ptr = std::shared_ptr<Map>;
  using MPKDADT = PointCloudAdaptor<std::vector<MapPoint<KeyFrame>::Ptr>>;
  using MPKDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
      nanoflann::L2_Simple_Adaptor<float, MPKDADT>, MPKDADT, 3>;

  Map();

  Map(const Map &map) = delete;

  // Getters
  const std::set<std::shared_ptr<KeyFrame>> get_kfs() const;

  const std::vector<MapPoint<KeyFrame>::Ptr> get_mps() const;

  const MPKDTree &get_kd_tree() const {
    // TODO 此处应该拷贝一个
    return kd_tree_;
  }

  std::shared_ptr<KeyFrame> get_last_kf() const;

  [[deprecated("Use get_feat_index")]] const std::unordered_map<
      size_t, std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr>>
  get_cls_index() const;

  // Map point operations
  void DumpMapPoints(std::string_view filename);

  void AddMapPoint(MapPoint<KeyFrame>::Ptr mp);

  void EraseMapPoint(MapPoint<KeyFrame>::Ptr mp);

  MapPoint<KeyFrame>::Ptr GetMapPointPtr(uuids::uuid uuid);

  std::vector<MapPoint<KeyFrame>::Ptr> GetNeighbors(const Eigen::Vector3f &pos,
                                                    float radius);

  // Key frame operations
  void AddKeyFrame(std::shared_ptr<KeyFrame> kf);

  // Class index operations
  void UpdateClsIndex(MapPoint<KeyFrame>::Ptr mp,
                      std::array<size_t, MAPPOINT_TOP_K> old_cls);

  void UpdateDescriptor(MapPoint<KeyFrame>::Ptr mp);

  //  void UpdateIndex(std::vector<MapPoint<KeyFrame>::Ptr> mps);

 private:
  mutable std::shared_mutex map_mutex_;
  mutable std::shared_mutex cls_mutex_;

  std::set<std::shared_ptr<KeyFrame>> kfs_{};
  std::vector<MapPoint<KeyFrame>::Ptr> mps_{};
  [[deprecated("Use feat_index")]] std::unordered_map<
      size_t, std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr>>
      cls_index_{};

  FeatureDB feat_index_{32};
  MPKDADT mps_adt_{mps_};
  MPKDTree kd_tree_;
  std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr> uuid_index_;
};

class PersistentMap {
 public:
  using Ptr = std::shared_ptr<PersistentMap>;
  using MPKDADT = PointCloudAdaptor<std::vector<MapPoint<KeyFrame>::Ptr>>;
  using MPKDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
      nanoflann::L2_Simple_Adaptor<float, MPKDADT>, MPKDADT, 3>;

  explicit PersistentMap(std::vector<MapPoint<KeyFrame>::Ptr> map_points);

  explicit PersistentMap(std::string_view filename);

  PersistentMap() = default;

  PersistentMap(const PersistentMap &map) = delete;

  PersistentMap(PersistentMap &&map) = default;

  MapPoint<KeyFrame>::Ptr GetMapPointPtr(uuids::uuid uuid);

  const std::vector<MapPoint<KeyFrame>::Ptr> &get_mps() const { return mps_; }

  const MPKDTree &get_kd_tree() const { return kd_tree_; }

  [[deprecated("Use get_feat_index")]] const std::unordered_map<
      size_t, std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr>>
      &get_cls_index() const {
    return cls_index_;
  }

  bool get_is_loaded() const { return is_loaded_; }

 private:
  std::vector<MapPoint<KeyFrame>::Ptr> mps_{};
  [[deprecated("Use feat_index")]] std::unordered_map<
      size_t, std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr>>
      cls_index_{};
  MPKDADT mps_adt_{mps_};
  MPKDTree kd_tree_;
  std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr> uuid_index_;
  bool is_loaded_ = false;
};

}  // namespace rgor

#endif  // RGOR_MAP_H