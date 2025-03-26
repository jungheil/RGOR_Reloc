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

#ifndef RGOR_GLOBAL_H
#define RGOR_GLOBAL_H

#include <uuid.h>

#include <Eigen/Geometry>
#include <memory>
#include <unordered_map>

#include "common/NeoMap.h"

namespace rgor {
// TODO: 同步NeoMap的uuid

class GBSubMap {
 public:
  using Ptr = std::shared_ptr<GBSubMap>;
  using RelocInfo = std::pair<std::pair<uuids::uuid, uuids::uuid>,
                              std::pair<Eigen::Vector4f, Eigen::Vector3f>>;
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
  explicit GBSubMap(NeoMap::Ptr map) {
    uuid_ = map->get_uuid();

    local_maps_[uuid_] = std::make_shared<NeoMap>(uuid_);
    UpdateMap(map);

    for (auto [_, mp] : map->get_mps()) {
      mp->set_gb_pos(mp->get_pos());
      mp->set_gb_init(true);
    }
    for (auto [_, kf] : map->get_kfs()) {
      kf->set_gb_init(true);
      kf->set_gb_r_cw(kf->get_abs_r_cw());
      kf->set_gb_t_cw(kf->get_abs_t_cw());
    }
  }

  const std::chrono::system_clock::time_point get_created_at() {
    return created_at_;
  }

  // get local map
  const std::unordered_map<uuids::uuid, NeoMap::Ptr> &get_local_maps() {
    return local_maps_;
  }

  // get reloc info
  const std::unordered_map<
      uuids::uuid, std::pair<std::pair<uuids::uuid, uuids::uuid>, RelocInfo>> &
  get_reloc_info() {
    return reloc_info_;
  }

  const std::unordered_map<uuids::uuid, uuids::uuid> &get_map_reloc_dict() {
    return map_reloc_dict_;
  }

 public:
  NeoMap::Ptr GetMapByUUID(uuids::uuid uuid) {
    if (local_maps_.find(uuid) == local_maps_.end()) {
      throw DataNotFoundException(uuid);
    }
    return local_maps_[uuid];
  }
  void UpdateMap(NeoMap::Ptr target_map) {
    if (local_maps_.find(target_map->get_uuid()) == local_maps_.end()) {
      throw DataNotFoundException(target_map->get_uuid());
    }
    auto base_map = local_maps_[target_map->get_uuid()];
    _SyncLocalMap(base_map, target_map);
    if (local_maps_.size() > 1) {
      _UpdateGB(base_map);
    }
  }

  void AddClose(const RelocInfo &reloc_info, uuids::uuid map_uuid_1,
                uuids::uuid map_uuid_2);

  void AddMap(GBSubMap::Ptr map, const RelocInfo &reloc_info,
              uuids::uuid map_uuid_1, uuids::uuid map_uuid_2) {
    if (local_maps_.find(map_uuid_1) != local_maps_.end()) {
      throw DataExistsException(map_uuid_1);
    }
    if (map->get_local_maps().find(map_uuid_2) != map->get_local_maps().end()) {
      throw DataExistsException(map_uuid_2);
    }
    auto [kf_uuids, T] = reloc_info;
    auto [kf_uuid_1, kf_uuid_2] = kf_uuids;
    auto [reloc_r, reloc_t] = T;

    auto map_base = local_maps_[map_uuid_1];
    auto map_merge = map->GetMapByUUID(map_uuid_2);

    auto base_kf = map_base->GetKFByUUID(kf_uuid_1);
    auto merge_kf = map_merge->GetKFByUUID(kf_uuid_2);

    // reloc 是merge关键帧在base关键帧下的相对位姿，根据base关键帧所在 gb的 r
    // t，以及 merge 关键帧在其所在 gb的 r t， 得到 merge 地图的 gb  到 base gb
    // 的变换 T，遍历所有merge关键帧，更新关键帧到 base的 gb r t

    // Get the global pose of the base keyframe
    Eigen::Quaternionf base_kf_gb_r_cw(base_kf->get_gb_r_cw());
    Eigen::Vector3f base_kf_gb_t_cw = base_kf->get_gb_t_cw();

    // Get the global pose of the merge keyframe in its own map
    Eigen::Quaternionf merge_kf_gb_r_cw(merge_kf->get_gb_r_cw());
    Eigen::Vector3f merge_kf_gb_t_cw = merge_kf->get_gb_t_cw();

    // Get the relative pose from base keyframe to merge keyframe
    Eigen::Quaternionf reloc_r_q(reloc_r);
    Eigen::Vector3f reloc_t_v = reloc_t;

    // Calculate the transformation from merge map's global coordinate system to
    // base map's global coordinate system
    Eigen::Quaternionf T_r =
        base_kf_gb_r_cw * reloc_r_q * merge_kf_gb_r_cw.inverse();
    Eigen::Vector3f T_t =
        base_kf_gb_t_cw + base_kf_gb_r_cw * reloc_t_v - T_r * merge_kf_gb_t_cw;

    for (auto [_, m] : map->get_local_maps()) {
      // Add the merge map to local_maps_
      local_maps_[m->get_uuid()] = m;

      // Update all keyframes in the merge map to the base map's global
      // coordinate system
      for (auto [_, kf] : m->get_kfs()) {
        kf->set_gb_r_cw((T_r * Eigen::Quaternionf(kf->get_gb_r_cw())).coeffs());
        kf->set_gb_t_cw(T_r * kf->get_gb_t_cw() + T_t);
        kf->set_gb_init(true);
      }

      for (auto [_, mp] : m->get_mps()) {
        mp->set_gb_pos(T_r * mp->get_pos() + T_t);
        mp->set_gb_init(true);
      }
    }

    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 engine(seq);
    uuids::uuid_random_generator gen(&engine);
    auto reloc_uuid = gen();

    map_reloc_dict_[map_uuid_1] = reloc_uuid;
    map_reloc_dict_[map_merge->get_uuid()] = reloc_uuid;
    reloc_info_[reloc_uuid] = {{map_uuid_1, map_uuid_2}, reloc_info};

    reloc_info_.insert(map->get_reloc_info().begin(),
                       map->get_reloc_info().end());
    map_reloc_dict_.insert(map->get_map_reloc_dict().begin(),
                           map->get_map_reloc_dict().end());
  }

 private:
  // 更新全局地图中新增或修改关键帧及其对应路标点的信息
  void _UpdateGB(NeoMap::Ptr base) {
    // auto updated_kfs = base->get_updated_kfs_set();
    // _UpdateBGKF(base, updated_kfs);

    auto new_kfs = base->get_new_kfs_set();

    _UpdateGBKF(base, new_kfs);
    _UpdateGBMP(base, new_kfs);
  }
  void _UpdateGBKF(NeoMap::Ptr base, std::unordered_set<uuids::uuid> kfs_uuid) {
    if (kfs_uuid.empty()) {
      return;
    }
    uuids::uuid kf_uuid;
    std::chrono::system_clock::time_point earliest_time =
        std::chrono::system_clock::now();
    for (const auto &uuid : kfs_uuid) {
      try {
        if (base->GetKFByUUID(uuid)->get_updated_at() < earliest_time) {
          kf_uuid = uuid;
          earliest_time = base->GetKFByUUID(uuid)->get_updated_at();
        }
      } catch (NeoMap::DataNotFoundException &e) {
        std::cout << e.what() << ", continue." << std::endl;
        continue;
      }
    }

    while (!kf_uuid.is_nil()) {
      try {
        auto kf = base->GetKFByUUID(kf_uuid);
        auto rel_r_cw = Eigen::Quaternionf(kf->get_rel_r_cw());
        auto rel_t_cw = kf->get_rel_t_cw();
        auto pkf = base->GetKFByUUID(kf->get_pre_kf());
        auto base_r_cw = Eigen::Quaternionf(pkf->get_gb_r_cw());
        auto base_t_cw = pkf->get_gb_t_cw();

        // Calculate global pose for current keyframe
        Eigen::Quaternionf gb_r_cw = base_r_cw * rel_r_cw;
        Eigen::Vector3f gb_t_cw = base_t_cw + base_r_cw * rel_t_cw;

        // Update the keyframe's global pose
        kf->set_gb_r_cw(gb_r_cw.coeffs());
        kf->set_gb_t_cw(gb_t_cw);
        kf->set_gb_init(true);

        // Move to next keyframe in the update queue
        base->remove_updated_kfs_set(kf_uuid);
        kf_uuid = kf->get_next_kf();
      } catch (NeoMap::DataNotFoundException &e) {
        std::cout << e.what() << ", break." << std::endl;
        break;
      }
    }
  }

  void _UpdateGBMP(NeoMap::Ptr base, std::unordered_set<uuids::uuid> kfs_uuid) {
    if (!kfs_uuid.empty()) {
      return;
    }
    std::unordered_set<uuids::uuid> updated_mps;
    for (const auto &uuid : kfs_uuid) {
      try {
        auto kf = base->GetKFByUUID(uuid);
        auto q_kf_cw = Eigen::Quaternionf(kf->get_abs_r_cw());
        auto t_kf_cw = kf->get_abs_t_cw();

        for (const auto [mp_id, _] : kf->get_measurement_mps()) {
          if (updated_mps.find(mp_id) == updated_mps.end()) {
            continue;
          }
          auto base_mp = base->GetMPByUUID(uuid);
          auto p_mp_w = base_mp->get_pos();
          auto p_mp_c = q_kf_cw.inverse() * (p_mp_w - t_kf_cw);

          auto r_kf_gb_cw = Eigen::Quaternionf(kf->get_gb_r_cw());
          auto t_kf_gb_cw = kf->get_gb_t_cw();

          auto p_mp_gb_w = r_kf_gb_cw * p_mp_c + t_kf_gb_cw;

          base_mp->set_gb_pos(p_mp_gb_w);
          updated_mps.insert(mp_id);
        }
      } catch (NeoMap::DataNotFoundException &e) {
        std::cout << e.what() << std::endl;
      }
    }
  }
  void _SyncLocalMap(NeoMap::Ptr base, NeoMap::Ptr target) {
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
        base->AddKeyFrame(
            uuid, kf->get_rel_r_cw(), kf->get_rel_t_cw(), kf->get_abs_r_cw(),
            kf->get_abs_t_cw(), kf->get_measurement_mps(), kf->get_pre_kf(),
            uuids::uuid(), kf->get_updated_at(), kf->get_created_at(), true);
        base->UpdateKeyFrame(kf->get_pre_kf(), std::nullopt, std::nullopt,
                             std::nullopt, std::nullopt, std::nullopt,
                             std::nullopt, uuid);
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

        auto old_rel_r_cw = base->GetKFByUUID(uuid)->get_rel_r_cw();
        auto old_rel_t_cw = base->GetKFByUUID(uuid)->get_rel_t_cw();

        base->UpdateKeyFrame(uuid, kf->get_rel_r_cw(), kf->get_rel_t_cw(),
                             kf->get_abs_r_cw(), kf->get_abs_t_cw(),
                             kf->get_measurement_mps(), kf->get_pre_kf(),
                             kf->get_next_kf());

        if (old_rel_r_cw != kf->get_rel_r_cw() ||
            old_rel_t_cw != kf->get_rel_t_cw()) {
          base->insert_updated_kfs_set(uuid);
        }
      } catch (NeoMap::DataNotFoundException &e) {
        std::cerr << e.what() << std::endl;
      }
    }
  }

 private:
  uuids::uuid uuid_;

  std::unordered_map<uuids::uuid, NeoMap::Ptr> local_maps_;

  // reloc_uuid -> info
  std::unordered_map<uuids::uuid,
                     std::pair<std::pair<uuids::uuid, uuids::uuid>, RelocInfo>>
      reloc_info_;
  // map uuid -> reloc_uuid
  std::unordered_map<uuids::uuid, uuids::uuid> map_reloc_dict_;

  std::chrono::system_clock::time_point created_at_ =
      std::chrono::system_clock::now();
};

class GBMap {
 public:
  // (kf_uuid_1, kf_uuid_2) -> (reloc_r, reloc_t)
  using RelocInfo = std::pair<std::pair<uuids::uuid, uuids::uuid>,
                              std::pair<Eigen::Vector4f, Eigen::Vector3f>>;

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
  void Get(NeoMap::Ptr target_map) {
    auto uuid = target_map->get_uuid();
    SyncMap(target_map);
    auto reloc_v = GetReloc(uuid);
    if (reloc_v.empty()) {
      return;
    }
    for (auto [tmap_uuid, reloc] : reloc_v) {
      AddConstraint(reloc, uuid, tmap_uuid);
    }
  }

 private:
  std::vector<std::pair<uuids::uuid, GBMap::RelocInfo>> GetReloc(
      uuids::uuid local_uuid);

  std::pair<uuids::uuid, uuids::uuid> GetKFPair(
      NeoMap::Ptr local_map, NeoMap::Ptr target_map,
      std::vector<std::pair<uuids::uuid, uuids::uuid>> match) {
    std::set<uuids::uuid> mps_local;
    std::set<uuids::uuid> mps_target;
    for (auto [local, target] : match) {
      mps_local.insert(local);
      mps_target.insert(target);
    }

    std::set<uuids::uuid> kfs_local;
    for (auto uuid : mps_local) {
      kfs_local.insert(local_map->GetMPByUUID(uuid)->get_observations().begin(),
                       local_map->GetMPByUUID(uuid)->get_observations().end());
    }
    if (kfs_local.empty()) {
      throw DataNotFoundException(local_map->get_uuid());
    }

    size_t max_local_obs = 0;
    uuids::uuid max_kf_local;
    for (auto uuid : kfs_local) {
      std::set<uuids::uuid> inter;
      std::set<uuids::uuid> obs;
      for (auto [mp, kf] :
           local_map->GetKFByUUID(uuid)->get_measurement_mps()) {
        obs.insert(mp);
      }
      std::set_intersection(obs.begin(), obs.end(), mps_local.begin(),
                            mps_local.end(), std::inserter(inter, inter.end()));
      if (inter.size() > max_local_obs) {
        max_local_obs = inter.size();
        max_kf_local = uuid;
      }
    }

    std::set<uuids::uuid> kfs_target;
    for (auto uuid : kfs_target) {
      kfs_target.insert(
          target_map->GetMPByUUID(uuid)->get_observations().begin(),
          target_map->GetMPByUUID(uuid)->get_observations().end());
    }
    if (kfs_target.empty()) {
      throw DataNotFoundException(target_map->get_uuid());
    }

    size_t max_target_obs = 0;
    uuids::uuid max_kf_target;
    for (auto uuid : kfs_target) {
      std::set<uuids::uuid> inter;
      std::set<uuids::uuid> obs;
      for (auto [mp, kf] :
           target_map->GetKFByUUID(uuid)->get_measurement_mps()) {
        obs.insert(mp);
      }
      std::set_intersection(obs.begin(), obs.end(), mps_target.begin(),
                            mps_target.end(),
                            std::inserter(inter, inter.end()));
      if (inter.size() > max_target_obs) {
        max_target_obs = inter.size();
        max_kf_target = uuid;
      }
    }

    return {max_kf_local, max_kf_target};
  }

  void MergeMap(RelocInfo reloc_info, uuids::uuid map_uuid_1,
                uuids::uuid map_uuid_2) {
    auto [kf_uuids, T] = reloc_info;
    auto [reloc_r, reloc_t] = T;

    auto map_base = sub_maps_[local2sub_[map_uuid_1]];
    auto map_merge = sub_maps_[local2sub_[map_uuid_2]];

    if (map_base->get_created_at() > map_merge->get_created_at()) {
      std::swap(map_base, map_merge);
      reloc_info.first = {kf_uuids.second, kf_uuids.first};
      reloc_info.second = {Eigen::Quaternionf(reloc_r).inverse().coeffs(),
                           -reloc_t};
    }

    map_base->AddMap(map_merge, reloc_info, map_uuid_1, map_uuid_2);
    sub_maps_.erase(local2sub_[map_uuid_2]);
    local2sub_.erase(map_uuid_2);
  }

  void AddConstraint(RelocInfo reloc_info, uuids::uuid map_uuid_1,
                     uuids::uuid map_uuid_2) {
    if (local2sub_.find(map_uuid_1) == local2sub_.end()) {
      throw DataNotFoundException(map_uuid_1);
    }
    if (local2sub_.find(map_uuid_2) == local2sub_.end()) {
      throw DataNotFoundException(map_uuid_2);
    }
    auto sub_map_uuid_1 = local2sub_[map_uuid_1];
    auto sub_map_uuid_2 = local2sub_[map_uuid_2];
    if (sub_map_uuid_1 == sub_map_uuid_2) {
      sub_maps_[sub_map_uuid_1]->AddClose(reloc_info, map_uuid_1, map_uuid_2);
    } else {
      MergeMap(reloc_info, map_uuid_1, map_uuid_2);
    }
  }

  void AddMap(NeoMap::Ptr map) {
    if (local2sub_.find(map->get_uuid()) != local2sub_.end()) {
      throw DataExistsException(map->get_uuid());
    }
    sub_maps_[map->get_uuid()] = std::make_shared<GBSubMap>(map);
    local2sub_[map->get_uuid()] = map->get_uuid();
  }

  void UpdateMap(NeoMap::Ptr target_map) {
    if (local2sub_.find(target_map->get_uuid()) == local2sub_.end()) {
      throw DataNotFoundException(target_map->get_uuid());
    }

    auto sub_map = sub_maps_[local2sub_[target_map->get_uuid()]];

    sub_map->UpdateMap(target_map);
  }

  void SyncMap(NeoMap::Ptr target_map) {
    if (local2sub_.find(target_map->get_uuid()) == local2sub_.end()) {
      AddMap(target_map);
    } else {
      UpdateMap(target_map);
    }
  }

  //  private:
  //  void InitGB(uuids::uuid map_uuid_1, uuids::uuid map_uuid_2,
  //  Eigen::Vector4f reloc_r, Eigen::Vector3f reloc_t){
  //    NeoMap::Ptr base_map, merge_map;
  //    try{
  //      base_map = source_maps_.at(map_uuid_1);
  //      merge_map = source_maps_.at(map_uuid_2);
  //      if (base_map->get_created_at() > merge_map->get_created_at()){
  //        std::swap(base_map, merge_map);
  //      }
  //    } catch (DataNotFoundException &e) {
  //      std::cerr << e.what() << std::endl;
  //      return;
  //    }
  //  }

 private:
  std::unordered_map<uuids::uuid, GBSubMap::Ptr> sub_maps_;

  std::unordered_map<uuids::uuid, uuids::uuid> local2sub_;

  //  std::unordered_map<uuids::uuid, NeoMap::Ptr> source_maps_;
  //  // reloc_uuid -> info
  //  std::unordered_map<uuids::uuid, RelocInfo> reloc_info_;
  //  // map uuid -> reloc_uuid
  //  std::unordered_map<uuids::uuid, uuids::uuid> map_reloc_dict_;
  //  // uni_map_uuid -> vector<map_uuid>
  //  std::unordered_map<uuids::uuid, std::vector<uuids::uuid>> uni_maps_;
};

}  // namespace rgor

#endif  // RGOR_GLOBAL_H
