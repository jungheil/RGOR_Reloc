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

#include "Global.h"

#include "NeoRelocation.h"
#include "Optimizer.h"
#include "common/Params.h"

namespace rgor {

void GBSubMap::AddClose(const RelocInfo &reloc_info, uuids::uuid map_uuid_1,
              uuids::uuid map_uuid_2) {
  if (local_maps_.find(map_uuid_1) != local_maps_.end()) {
    throw DataExistsException(map_uuid_1);
  }
  if (local_maps_.find(map_uuid_2) != local_maps_.end()) {
    throw DataExistsException(map_uuid_2);
  }

  std::random_device rd;
  auto seed_data = std::array<int, std::mt19937::state_size>{};
  std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  std::mt19937 engine(seq);
  uuids::uuid_random_generator gen(&engine);
  auto reloc_uuid = gen();
  map_reloc_dict_[map_uuid_1] = reloc_uuid;
  map_reloc_dict_[map_uuid_2] = reloc_uuid;
  reloc_info_[reloc_uuid] = {{map_uuid_1, map_uuid_2}, reloc_info};

  OptimizeGBSubMap(this, reloc_info.first.first);
  GlobalBundleAdjustment(this, reloc_info.first.first);
}

std::vector<std::pair<uuids::uuid, GBMap::RelocInfo>> GBMap::GetReloc(
    uuids::uuid local_uuid) {
  std::vector<std::pair<uuids::uuid, RelocInfo>> ret;
  if (local2sub_.find(local_uuid) == local2sub_.end()) {
    throw DataNotFoundException(local_uuid);
  }
  auto sub_uuid = local2sub_[local_uuid];
  if (sub_maps_.find(local2sub_[sub_uuid]) == sub_maps_.end()) {
    throw DataNotFoundException(sub_uuid);
  }
  auto local_map = sub_maps_[sub_uuid]->GetMapByUUID(local2sub_[local_uuid]);
  // TODO arg
  if (local_map->get_new_mp_count() < 32) {
    return ret;
  }

  std::vector<uuids::uuid> selector = local_map->GetHotMPs();

  for (auto &[sub_uuid, sub_map] : sub_maps_) {
    for (auto &[target_uuid, target_map] : sub_map->get_local_maps()) {
      if (local_uuid == target_uuid) {
        continue;
      }
      auto [mp_pairs, r, t] = Relocation(RelocationParams())
                                  .Get<NeoMap, std::shared_ptr<NeoMapPoint>>(
                                      local_map, target_map, selector);
      if (mp_pairs.size() > 0) {
        auto kf_pairs = GetKFPair(local_map, target_map, mp_pairs);
        ret.push_back({target_uuid, {kf_pairs, {r, t}}});
      }
    }
  }

  if (ret.size() > 0) {
    local_map->ClearHotMPs();
  }

  return ret;
}
}