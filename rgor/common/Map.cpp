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

#include "common/Map.h"

#include <cstddef>
#include <fstream>

#include "proto/PMap.pb.h"

namespace rgor {

// Map class implementations
Map::Map() : kd_tree_(3, mps_adt_) {}

const std::set<std::shared_ptr<KeyFrame>> Map::get_kfs() const {
  std::shared_lock lock(map_mutex_);
  return kfs_;
}

// TODO 不安全，尤其是&，除非至少要浅拷贝vector
const std::vector<MapPoint<KeyFrame>::Ptr> Map::get_mps() const {
  std::shared_lock lock(map_mutex_);
  return mps_;
}

const std::unordered_map<
    size_t, std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr>>
Map::get_cls_index() const {
  std::shared_lock lock(cls_mutex_);
  return cls_index_;
}

void Map::AddMapPoint(MapPoint<KeyFrame>::Ptr mp) {
  std::unique_lock map_lock(map_mutex_);
  std::unique_lock cls_lock(cls_mutex_);

  if (!mp) {
    throw std::invalid_argument("Null map point");
  }

  mps_.push_back(mp);
  mp->set_on_map(true);
  mp->set_map_id(mps_.size() - 1);
  kd_tree_.addPoints(mps_.size() - 1, mps_.size() - 1);

  for (auto c : mp->get_cls()) {
    cls_index_[c].insert({mp->get_uuid(), mp});
  }
  uuid_index_[mp->get_uuid()] = mp;
}

void Map::EraseMapPoint(MapPoint<KeyFrame>::Ptr mp) {
  assert(mps_[mp->get_map_id()] == mp);
  if (mp != mps_.back()) {
    for (size_t i : mp->get_cls()) {
      cls_index_[i].erase(mp->get_uuid());
    }
    mp->set_on_map(false);
    // std::swap(mps_[mp->get_map_id()], mps_.back());
    // mps_.pop_back();
    kd_tree_.removePoint(mp->get_map_id());
  }
  uuid_index_.erase(mp->get_uuid());
}

MapPoint<KeyFrame>::Ptr Map::GetMapPointPtr(uuids::uuid uuid) {
  if (uuid_index_.find(uuid) != uuid_index_.end()) {
    return uuid_index_[uuid];
  }
  return nullptr;
}

std::vector<MapPoint<KeyFrame>::Ptr>
Map::GetNeighbors(const Eigen::Vector3f &pos, float radius) {
  std::vector<MapPoint<KeyFrame>::Ptr> ret;
  std::vector<size_t> indices;
  std::vector<float> dists;
  float query_pt[3] = {pos[0], pos[1], pos[2]};
  std::vector<nanoflann::ResultItem<size_t, float>> indices_dists;
  nanoflann::RadiusResultSet<float, size_t> result_set(radius * radius,
                                                       indices_dists);

  kd_tree_.findNeighbors(result_set, query_pt);

  for (auto &item : indices_dists) {
    ret.push_back(mps_[item.first]);
  }
  return ret;
}

void Map::AddKeyFrame(std::shared_ptr<KeyFrame> kf) { kfs_.insert(kf); }

std::shared_ptr<KeyFrame> Map::get_last_kf() const {
  std::shared_ptr<KeyFrame> ret = nullptr;
  for (auto itr = kfs_.rbegin(); itr != kfs_.rend(); ++itr) {
    if ((*itr) != nullptr) {
      ret = *itr;
      break;
    }
  }
  return ret;
}

void Map::UpdateClsIndex(MapPoint<KeyFrame>::Ptr mp,
                         std::array<size_t, MAPPOINT_TOP_K> old_cls) {
  // XXX 此处可以优化忽略一致的
  for (size_t i : old_cls) {
    cls_index_[i].erase(mp->get_uuid());
  }
  for (size_t i : mp->get_cls()) {
    cls_index_[i].insert({mp->get_uuid(), mp});
  }
}

// PersistentMap class implementations
PersistentMap::PersistentMap(std::vector<MapPoint<KeyFrame>::Ptr> map_points)
    : kd_tree_(3, mps_adt_) {
  is_loaded_ = true;
  for (auto mp : map_points) {
    auto new_mp = std::make_shared<MapPoint<KeyFrame>>(
        mp->get_pos(), mp->get_scale(), mp->get_descriptor());
    new_mp->set_on_map(true);
    new_mp->set_map_id(mps_.size() - 1);
    mps_.push_back(new_mp);
    kd_tree_.addPoints(mps_.size() - 1, mps_.size() - 1);
    uuid_index_[new_mp->get_uuid()] = new_mp;
    for (auto c : new_mp->get_cls()) {
      cls_index_[c].insert(
          {mps_[mps_.size() - 1]->get_uuid(), mps_[mps_.size() - 1]});
    }
  }
}

PersistentMap::PersistentMap(std::string_view filename)
    : kd_tree_(3, mps_adt_) {
  if (filename.empty()) {
    return;
  }
  PMap pmap;
  std::ifstream ifs(filename.data(), std::ios::in | std::ios::binary);
  if (!pmap.ParseFromIstream(&ifs)) {
    std::cerr << "Failed to read map points from " << filename << std::endl;
    return;
  }
  is_loaded_ = true;
  for (const auto &point : pmap.points()) {
    Eigen::Vector3f pos(point.pose().x(), point.pose().y(), point.pose().z());
    std::pair<float, float> scale(point.scale().s(), point.scale().l());
    Eigen::VectorXf descriptor;
    descriptor.resize(point.desc().data_size());
    for (int i = 0; i < point.desc().data_size(); ++i) {
      descriptor[i] = point.desc().data(i);
    }
    auto mp = std::make_shared<MapPoint<KeyFrame>>(pos, scale, descriptor);
    mps_.push_back(mp);
    mp->set_on_map(true);
    mp->set_map_id(mps_.size() - 1);
    kd_tree_.addPoints(mps_.size() - 1, mps_.size() - 1);
    for (auto c : mp->get_cls()) {
      cls_index_[c].insert({mp->get_uuid(), mp});
    }
  }
  std::cout << "Loaded " << mps_.size() << " map points from " << filename
            << std::endl;
}

MapPoint<KeyFrame>::Ptr PersistentMap::GetMapPointPtr(uuids::uuid uuid) {
  if (uuid_index_.find(uuid) != uuid_index_.end()) {
    return uuid_index_[uuid];
  }
  return nullptr;
}

void Map::DumpMapPoints(std::string_view filename) {
  PMap pmap;
  std::shared_lock lock(map_mutex_);
  for (auto mp : mps_) {
    if (mp == nullptr || !mp->get_on_map() || mp->get_bad()) {
      continue;
    }
    PMapPoint *point = pmap.add_points();
    PMPPose *pose = point->mutable_pose();
    pose->set_x(mp->get_pos()[0]);
    pose->set_y(mp->get_pos()[1]);
    pose->set_z(mp->get_pos()[2]);

    PMPScale *scale = point->mutable_scale();
    scale->set_s(mp->get_scale().first);
    scale->set_l(mp->get_scale().second);

    PMPDescriptor *desc = point->mutable_desc();
    auto descriptor = mp->get_descriptor();
    for (size_t i = 0; i < descriptor.size(); ++i) {
      desc->add_data(descriptor[i]);
    }
  }

  std::ofstream ofs(filename.data(),
                    std::ios::out | std::ios::trunc | std::ios::binary);
  if (!pmap.SerializeToOstream(&ofs)) {
    std::cerr << "Failed to write map points to " << filename << std::endl;
    return;
  }
}

} // namespace rgor
