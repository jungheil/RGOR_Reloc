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

#ifndef RGOR_FRAME_H
#define RGOR_FRAME_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "uuid.h"

namespace rgor {

template <typename frame_type> class MapPoint;

struct KeyPoint {
  using Ptr = std::shared_ptr<KeyPoint>;

  KeyPoint(float x, float y, int depth, float w, float h,
           Eigen::VectorXf descriptor)
      : x(x), y(y), depth(depth), w(w), h(h),
        descriptor(std::move(descriptor)) {}

  float x, y;
  int depth;
  float w, h;
  Eigen::VectorXf descriptor;
};

struct Camera {
  using Ptr = std::shared_ptr<Camera>;

  Camera(Eigen::Matrix3f intrinsics, Eigen::VectorXf distortion, size_t width,
         size_t height, float min_depth, float max_depth, float valid_wh_ratio)
      : K(std::move(intrinsics)), D(std::move(distortion)), width(width),
        height(height), min_depth(min_depth), max_depth(max_depth),
        valid_wh_ratio(valid_wh_ratio), K_inv(K.inverse()) {
    assert(valid_wh_ratio <= 1 && valid_wh_ratio >= 0);
  }

  Eigen::Matrix3f K;
  Eigen::Matrix3f K_inv;
  Eigen::VectorXf D;
  size_t width, height;
  float min_depth, max_depth, valid_wh_ratio;
};

class Frame {
public:
  using Ptr = std::shared_ptr<Frame>;

  Frame(Camera::Ptr camera, Eigen::Vector4f r_cw, Eigen::Vector3f t_cw,
        time_t timestamp, std::vector<KeyPoint::Ptr> kps)
      : camera_(std::move(camera)), r_cw_(std::move(r_cw)),
        t_cw_(std::move(t_cw)), timestamp_(timestamp), kps_(std::move(kps)) {
    if (!camera_) {
      throw std::invalid_argument("Camera pointer cannot be null");
    }
    ResolveKP();

    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 engine(seq);
    uuids::uuid_random_generator gen(&engine);
    uuid_ = gen();
  }

  Frame(const Frame &) = delete;

  Frame &operator=(const Frame &) = delete;

  Frame(Frame &&other) noexcept
      : camera_(std::move(other.camera_)), r_cw_(std::move(other.r_cw_)),
        t_cw_(std::move(other.t_cw_)), timestamp_(other.timestamp_),
        kps_(std::move(other.kps_)),
        kp_position_(std::move(other.kp_position_)), uuid_(other.uuid_) {}

  Frame &operator=(Frame &&other) noexcept {
    if (this != &other) {
      camera_ = std::move(other.camera_);
      r_cw_ = std::move(other.r_cw_);
      t_cw_ = std::move(other.t_cw_);
      timestamp_ = other.timestamp_;
      kps_ = std::move(other.kps_);
      kp_position_ = std::move(other.kp_position_);
      uuid_ = other.uuid_;
    }
    return *this;
  }

  uuids::uuid get_uuid() const { return uuid_; }

  [[nodiscard]] const std::vector<KeyPoint::Ptr> &get_kps() const {
    return kps_;
  }

  [[nodiscard]] Camera::Ptr get_camera() const { return camera_; }

  [[nodiscard]] const Eigen::Vector4f &get_r_cw() const {
    std::shared_lock<std::shared_mutex> lock(mtx_);
    return r_cw_;
  }

  void set_r_cw(const Eigen::Vector4f &r_cw) {
    std::unique_lock<std::shared_mutex> lock(mtx_);
    r_cw_ = r_cw;
  }

  void set_r_cw(const Eigen::Matrix3f &R) {
    Eigen::Quaternionf q(R);
    std::unique_lock<std::shared_mutex> lock(mtx_);
    r_cw_ = q.coeffs();
  }

  [[nodiscard]] const Eigen::Vector3f &get_t_cw() const {
    std::shared_lock<std::shared_mutex> lock(mtx_);
    return t_cw_;
  }

  void set_t_cw(const Eigen::Vector3f &t_cw) {
    std::unique_lock<std::shared_mutex> lock(mtx_);
    t_cw_ = t_cw;
  }

  [[nodiscard]] const time_t get_timestamp() const { return timestamp_; }

  [[nodiscard]] const std::vector<Eigen::Vector3f> &get_kp_position() const {
    return kp_position_;
  }

private:
  bool IsGoodKP(KeyPoint::Ptr kp) const {
    if (kp->depth < camera_->min_depth * 1e3 ||
        kp->depth > camera_->max_depth * 1e3) {
      return false;
    }
    // if (kp->x < 0 || kp->x > camera_->width || kp->y < 0 ||
    //     kp->y > camera_->height) {
    //   return false;
    // }
    // if (kp->x < camera_->width * (1 - camera_->valid_wh_ratio) / 2 ||
    //     kp->x > camera_->width * (1 + camera_->valid_wh_ratio) / 2 ||
    //     kp->y < camera_->height * (1 - camera_->valid_wh_ratio) / 2 ||
    //     kp->y > camera_->height * (1 + camera_->valid_wh_ratio) / 2) {
    //   return false;
    // }
    if (kp->x - kp->w / 2 <
            camera_->width * (1 - camera_->valid_wh_ratio) / 2 ||
        kp->x + kp->w / 2 >
            camera_->width * (1 + camera_->valid_wh_ratio) / 2 ||
        kp->y - kp->h / 2 <
            camera_->height * (1 - camera_->valid_wh_ratio) / 2 ||
        kp->y + kp->h / 2 >
            camera_->height * (1 + camera_->valid_wh_ratio) / 2) {
      return false;
    }
    return true;
  }

  void ResolveKP() {
    std::vector<Eigen::Vector3f>(kps_.size(), Eigen::Vector3f::Zero())
        .swap(kp_position_);
    // XXX ignore distortion

    Eigen::Matrix3Xf pts_img(3, kps_.size());
    Eigen::VectorXf depth(kps_.size());

    for (size_t i = 0; i < kps_.size(); i++) {
      pts_img(0, i) = kps_[i]->x;
      pts_img(1, i) = kps_[i]->y;
      depth(i) = kps_[i]->depth * 1e-3;
    }
    pts_img.row(2) = Eigen::VectorXf::Ones(kps_.size());
    Eigen::Matrix3Xf pts_img_d =
        (pts_img.transpose().array().colwise() * depth.array()).transpose();

    Eigen::Matrix3Xf pts_cam = camera_->K_inv * pts_img_d;
    for (size_t i = 0; i < kps_.size(); i++) {
      if (IsGoodKP(kps_[i])) {
        Eigen::Vector3f pos_c = pts_cam.col(i);
        Eigen::Quaternionf q(r_cw_);
        auto R = q.toRotationMatrix();
        Eigen::Vector3f pos_w = R * pos_c + t_cw_;

        kp_position_[i] = pos_w;
      }
    }
  }

private:
  uuids::uuid uuid_;

  std::vector<KeyPoint::Ptr> kps_;
  Camera::Ptr camera_;

  Eigen::Vector4f r_cw_; // Rotation from world to camera
  Eigen::Vector3f t_cw_; // Translation from world to camera

  std::vector<Eigen::Vector3f> kp_position_;

  time_t timestamp_;

  mutable std::shared_mutex mtx_;
};

struct KFMeasurement {
  Eigen::Vector3f position;
  Eigen::VectorXf descriptor;
  std::pair<float, float> scale;
};

class KeyFrame {
public:
  using Ptr = std::shared_ptr<KeyFrame>;

  KeyFrame(Camera::Ptr camera, Eigen::Vector4f r_cw, Eigen::Vector3f t_cw,
           time_t timestamp,
           std::vector<std::shared_ptr<MapPoint<KeyFrame>>> mps,
           std::vector<KFMeasurement> measurement, bool fixed = false)
      : camera_(std::move(camera)), r_cw_(std::move(r_cw)),
        t_cw_(std::move(t_cw)), timestamp_(timestamp), mps_(std::move(mps)),
        measurement_(std::move(measurement)), fixed_(fixed) {
    id_ = ++id_counter_;

    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 engine(seq);
    uuids::uuid_random_generator gen(&engine);
    uuid_ = gen();
    // assert(camera != nullptr);
  }

  KeyFrame(const KeyFrame &) = delete;

  KeyFrame &operator=(const KeyFrame &) = delete;

  KeyFrame(KeyFrame &&other) noexcept
      : camera_(std::move(other.camera_)), r_cw_(std::move(other.r_cw_)),
        t_cw_(std::move(other.t_cw_)), timestamp_(other.timestamp_),
        mps_(std::move(other.mps_)),
        measurement_(std::move(other.measurement_)), fixed_(other.fixed_),
        id_(other.id_), uuid_(other.uuid_), bad_(other.bad_),
        on_map_(other.on_map_), parents_kf_(std::move(other.parents_kf_)),
        children_kf_(std::move(other.children_kf_)) {}

  KeyFrame &operator=(KeyFrame &&other) noexcept {
    if (this != &other) {
      camera_ = std::move(other.camera_);
      r_cw_ = std::move(other.r_cw_);
      t_cw_ = std::move(other.t_cw_);
      timestamp_ = other.timestamp_;
      mps_ = std::move(other.mps_);
      measurement_ = std::move(other.measurement_);
      fixed_ = other.fixed_;
      id_ = other.id_;
      uuid_ = other.uuid_;
      bad_ = other.bad_;
      on_map_ = other.on_map_;
      parents_kf_ = std::move(other.parents_kf_);
      children_kf_ = std::move(other.children_kf_);
    }
    return *this;
  }

  size_t get_id() const { return id_; }

  uuids::uuid get_uuid() const { return uuid_; }

  [[nodiscard]] Camera::Ptr get_camera() const { return camera_; }

  [[nodiscard]] const Eigen::Vector4f &get_r_cw() {
    std::shared_lock<std::shared_mutex> lck(mtx_);
    return r_cw_;
  }

  void set_r_cw(const Eigen::Vector4f &r_cw) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    r_cw_ = r_cw;
  }

  void set_r_cw(const Eigen::Matrix3f &R) {
    Eigen::Quaternionf q(R);
    std::unique_lock<std::shared_mutex> lck(mtx_);
    r_cw_ = q.coeffs();
  }

  [[nodiscard]] const Eigen::Vector3f &get_t_cw() {
    std::shared_lock<std::shared_mutex> lck(mtx_);
    return t_cw_;
  }

  void set_t_cw(const Eigen::Vector3f &t_cw) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    t_cw_ = t_cw;
  }

  [[nodiscard]] const time_t get_timestamp() const { return timestamp_; }

  [[nodiscard]] const std::vector<KFMeasurement> &get_measurement() const {
    return measurement_;
  }

  [[nodiscard]] const std::vector<std::shared_ptr<MapPoint<KeyFrame>>>
  get_mps() const {
    return mps_;
  }

  [[nodiscard]] const bool get_bad() const { return bad_; }

  void set_bad(bool bad) { bad_ = bad; }

  void set_on_map(bool on_map) { on_map_ = on_map; }

  const bool get_on_map() const { return on_map_; }

  void set_fixed(bool fixed) { fixed_ = fixed; }

  const bool get_fixed() const { return fixed_; }

  void set_parents_kf(KeyFrame::Ptr parents_kf) { parents_kf_ = parents_kf; }

  const KeyFrame::Ptr get_parents_kf() const { return parents_kf_; }

  void set_children_kf(KeyFrame::Ptr children_kf) {
    children_kf_ = children_kf;
  }

  const KeyFrame::Ptr get_children_kf() const { return children_kf_; }

private:
  void AddMPS(const std::vector<std::shared_ptr<MapPoint<KeyFrame>>> &mps,
              const std::vector<KFMeasurement> &measurement) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    mps_.insert(mps_.end(), mps.begin(), mps.end());
    measurement_.insert(measurement_.end(), measurement.begin(),
                        measurement.end());
  }

private:
  size_t id_ = 0;
  static std::atomic<size_t> id_counter_;
  uuids::uuid uuid_;

  Camera::Ptr camera_;

  Eigen::Vector4f r_cw_; // Rotation from world to camera
  Eigen::Vector3f t_cw_; // Translation from world to camera

  std::vector<std::shared_ptr<MapPoint<KeyFrame>>> mps_;
  std::vector<KFMeasurement> measurement_;

  bool bad_ = false;
  bool on_map_ = false;
  bool fixed_ = false;

  time_t timestamp_;

  KeyFrame::Ptr parents_kf_ = nullptr;
  KeyFrame::Ptr children_kf_ = nullptr;

  mutable std::shared_mutex mtx_;
};

} // namespace rgor

#endif // RGOR_FRAME_H
