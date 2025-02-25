//
// Created by cat on 25-2-23.
//

#ifndef RGOR_SYS_TEMP_H
#define RGOR_SYS_TEMP_H

#include <common/Frame.h>
#include <common/Map.h>
#include <common/NeoMap.h>

namespace rgor {
static NeoMap::Ptr map_to_neo_map(Map::Ptr map) {
  auto neo_map = std::make_shared<NeoMap>();

  // 转换地图点
  auto mps = map->get_mps();
  for (const auto& mp : mps) {
    if (mp == nullptr || mp->get_bad() || !mp->get_on_map()) continue;

    // 获取观测该点的关键帧UUID
    std::unordered_set<uuids::uuid> observations;
    auto obs = mp->get_observations();
    for (const auto& [uuid, _] : obs) {
      observations.insert(uuid);
    }

    // 添加地图点
    neo_map->AddMapPoint(mp->get_uuid(), mp->get_descriptor(), mp->get_pos(),
                         mp->get_scale(), observations);
  }

  // 转换关键帧
  auto kfs = map->get_kfs();
  // 第一遍添加所有关键帧
  for (const auto& kf : kfs) {
    if (kf == nullptr || kf->get_bad()) continue;

    // 获取关键帧观测到的地图点测量
    std::unordered_map<uuids::uuid, Eigen::Vector3f> measurements;
    auto meas = kf->get_measurement();
    auto mps = kf->get_mps();
    for (size_t i = 0; i < mps.size(); i++) {
      if (mps[i] == nullptr || mps[i]->get_bad() || !mps[i]->get_on_map())
        continue;
      measurements[mps[i]->get_uuid()] = meas[i].position;
    }

    // 添加关键帧
    neo_map->AddKeyFrame(kf->get_uuid(), kf->get_r_cw(), kf->get_t_cw(),
                         kf->get_r_cw(), kf->get_t_cw(), measurements);
  }

  // 第二遍设置关键帧的父子关系
  for (const auto& kf : kfs) {
    if (kf == nullptr || kf->get_bad()) continue;

    auto parent = kf->get_parents_kf();
    auto child = kf->get_children_kf();

    NeoKeyFrame::Ptr neo_kf;
    try {
      neo_kf = neo_map->GetKFByUUID(kf->get_uuid());

    } catch (const NeoMap::DataNotFoundException&) {
      std::cout << "add Pre & Next KF failed." << std::endl;
    }
    if (parent && !parent->get_bad()) {
      neo_kf->set_pre_kf(parent->get_uuid());
      auto neo_parent = neo_map->GetKFByUUID(parent->get_uuid());
      if (neo_parent) {
        Eigen::Quaternionf qp(parent->get_r_cw());
        Eigen::Quaternionf ql(kf->get_r_cw());
        Eigen::Quaternionf q = qp.conjugate() * ql;
        Eigen::Vector4f r = q.coeffs();

        auto t = kf->get_t_cw() - parent->get_t_cw();

        neo_kf->set_rel_r_cw(r);
        neo_kf->set_rel_t_cw(t);
      }
    }

    if (child && !child->get_bad()) {
      neo_kf->set_next_kf(child->get_uuid());
    }
  }

  return neo_map;
}
}  // namespace rgor

#endif  // RGOR_SYS_TEMP_H
