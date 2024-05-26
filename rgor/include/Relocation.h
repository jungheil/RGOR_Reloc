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

#ifndef RGOR_RELOCATION_H
#define RGOR_RELOCATION_H

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Optimizer.h"
#include "common/Frame.h"
#include "common/Map.h"
#include "utils/KMAssignment.h"
#include "utils/utils.h"

namespace rgor {

template <typename MP>
std::vector<MP> GetBestMatchGroup(
    const std::vector<MP> &query_group,
    const std::unordered_map<uuids::uuid, std::vector<MP>> query_neigh_dict) {
  std::unordered_map<uuids::uuid, size_t> group_dict;
  std::vector<std::unordered_set<MP>> groups;
  for (size_t i = 0; i < query_group.size(); ++i) {
    if (query_group[i] == nullptr) {
      continue;
    }
    if (group_dict.find(query_group[i]->get_uuid()) == group_dict.end()) {
      group_dict[query_group[i]->get_uuid()] = groups.size();
      groups.push_back({});
    }
    groups[group_dict[query_group[i]->get_uuid()]].insert(query_group[i]);

    for (auto item :
         query_neigh_dict.find(query_group[i]->get_uuid())->second) {
      if (group_dict.find(item->get_uuid()) == group_dict.end()) {
        group_dict[item->get_uuid()] = group_dict[query_group[i]->get_uuid()];
        // groups[group_dict[query_group[i]->get_uuid()]].insert(item);
      }
    }
  }
  std::vector<MP> ret;
  size_t max_group_size = 0;
  size_t max_group_idx = 0;
  for (size_t i = 0; i < groups.size(); ++i) {
    if (groups[i].size() > max_group_size) {
      max_group_size = groups[i].size();
      max_group_idx = i;
    }
  }
  ret = std::vector<MP>(groups[max_group_idx].begin(),
                        groups[max_group_idx].end());
  return ret;
}

template <typename MP>
std::tuple<std::vector<MP>, std::vector<MP>, Eigen::Vector4f>
GetBestMatch(const std::vector<MP> &mps_1, const std::vector<MP> &mps_2) {
  assert(mps_1.size() == mps_2.size());
  size_t points_size = mps_1.size();
  std::vector<size_t> ret_idx(points_size);
  for (size_t i = 0; i < points_size; ++i) {
    ret_idx[i] = i;
  }
  std::vector<std::vector<float>> dist_matrix(
      points_size, std::vector<float>(points_size, 0));
  std::vector<std::vector<Eigen::Quaternionf>> q_matrix(
      points_size, std::vector<Eigen::Quaternionf>(points_size));
  for (size_t i = 0; i < points_size; ++i) {
    for (size_t j = i + 1; j < points_size; ++j) {
      float dist_1 = (mps_1[i]->get_pos() - mps_1[j]->get_pos()).norm();
      float dist_2 = (mps_2[i]->get_pos() - mps_2[j]->get_pos()).norm();
      dist_matrix[i][j] = std::abs(dist_1 - dist_2);
      dist_matrix[j][i] = std::abs(dist_1 - dist_2);
      Eigen::Quaternionf q;
      q.setFromTwoVectors(mps_1[j]->get_pos() - mps_1[i]->get_pos(),
                          mps_2[j]->get_pos() - mps_2[i]->get_pos());
      q_matrix[i][j] = q;
      // 这里应该不需要inv
      q_matrix[j][i] = q;
    }
  }

  size_t min_idx_1 = 0;
  size_t min_idx_2 = 0;
  while (points_size > 1) {
    std::vector<float> dist_mean(points_size);
    for (size_t i = 0; i < points_size; ++i) {
      float sum = 0;
      for (size_t j = 0; j < points_size; ++j) {
        sum += dist_matrix[i][j];
      }
      dist_mean[i] = sum / points_size;
    }

    size_t max_idx = 0;
    float max_dist = 0;
    float min_dist_1 = std::numeric_limits<float>::max();
    float min_dist_2 = std::numeric_limits<float>::max();
    for (size_t i = 0; i < points_size; ++i) {
      if (dist_mean[i] > max_dist) {
        max_dist = dist_mean[i];
        max_idx = i;
      }
      if (dist_mean[i] < min_dist_2) {
        min_dist_2 = dist_mean[i];
        min_idx_2 = i;
        if (min_dist_2 < min_dist_1) {
          size_t t_idx = min_idx_1;
          float t_dist = min_dist_1;
          min_idx_1 = min_idx_2;
          min_idx_2 = t_idx;
          min_dist_1 = min_dist_2;
          min_dist_2 = t_dist;
        }
      }
    }
    if (max_dist > 0.5) {
      dist_matrix.erase(dist_matrix.begin() + max_idx);
      for (auto itr = dist_matrix.begin(); itr != dist_matrix.end(); ++itr) {
        itr->erase(itr->begin() + max_idx);
      }
      q_matrix.erase(q_matrix.begin() + max_idx);
      for (auto itr = q_matrix.begin(); itr != q_matrix.end(); ++itr) {
        itr->erase(itr->begin() + max_idx);
      }
      ret_idx.erase(ret_idx.begin() + max_idx);
      points_size--;
    } else {
      break;
    }
  }

  // TODO 过滤异常点

  if (points_size < 2) {
    return {{}, {}, {}};
  }

  float total_err = std::numeric_limits<float>::max();
  size_t total_min_idx = 0;
  for (size_t i = 0; i < points_size; ++i) {
    float err = 0;
    for (size_t j = 0; j < points_size; ++j) {
      err += dist_matrix[i][j];
    }
    if (err < total_err) {
      total_err = err;
      total_min_idx = i;
    }
  }

  std::vector<MP> ret_1;
  std::vector<MP> ret_2;
  for (size_t i = 0; i < points_size; ++i) {
    ret_1.push_back(mps_1[ret_idx[i]]);
    ret_2.push_back(mps_2[ret_idx[i]]);
  }
  // min_idx_1 换到0
  std::swap(ret_1[0], ret_1[total_min_idx]);
  std::swap(ret_2[0], ret_2[total_min_idx]);
  std::swap(ret_idx[0], ret_idx[total_min_idx]);

  std::vector<Eigen::Vector3f> ret_1_pos;
  std::vector<Eigen::Vector3f> ret_2_pos;
  for (size_t i = 0; i < points_size; ++i) {
    ret_1_pos.push_back(ret_1[i]->get_pos());
    ret_2_pos.push_back(ret_2[i]->get_pos());
  }

  // TODO 获取平移旋转初值

  return {ret_1, ret_2, {0, 0, 0, 1}};
}

template <typename MP>
std::pair<float, std::vector<std::pair<MP, MP>>>
MatchByDesc(MP key, MP query, const std::vector<MP> &key_neigh,
            const std::vector<MP> &query_neigh) {
  std::vector<std::vector<float>> cost(
      key_neigh.size(), std::vector<float>(query_neigh.size(), 0));
  for (size_t i = 0; i < key_neigh.size(); ++i) {
    for (size_t j = 0; j < query_neigh.size(); ++j) {
      auto dist =
          abs(rgor::l2_distance(key_neigh[i]->get_pos(), key->get_pos()) -
              rgor::l2_distance(query_neigh[j]->get_pos(), query->get_pos()));
      // TODO ADD THRESHOLD ARG
      auto dist_score = (0.5 - dist) * 2;
      auto desc_socre =
          1 - rgor::corr_distance(key_neigh[i]->get_descriptor(),
                                  query_neigh[j]->get_descriptor());
      auto scale_score =
          GetScaleScore(key_neigh[i]->get_scale(), query_neigh[j]->get_scale());
      if (dist_score < 0 || desc_socre < 0.2 || scale_score < 0.5) {
        cost[i][j] = 1e-3;
      } else {
        cost[i][j] = dist_score + scale_score + desc_socre;
      }
    }
  }
  auto match = Assignment(cost);
  float score = 0;
  std::vector<std::pair<MP, MP>> ret;

  for (size_t i = 0; i < match.size(); ++i) {
    if (cost[match[i].first][match[i].second] > 1e-3) {
      score += cost[match[i].first][match[i].second];
      ret.push_back({key_neigh[match[i].first], query_neigh[match[i].second]});
    }
  }
  score /= ret.size();
  return {score, ret};
}

class Relocation {
public:
  Relocation(const RelocationParams &params)
      : neighbour_radius_(params.neighbour_radius),
        scale_score_threshold_(params.scale_score_threshold),
        desc_score_threshold_(params.desc_score_threshold),
        pair_score_threshold_(params.pair_score_threshold),
        best_match_threshold_(params.best_match_threshold),
        fine_dist_threshold_(params.fine_dist_threshold),
        fine_desc_threshold_(params.fine_desc_threshold){};

  // TODO 应该用只包含位置和描述子的数据结构或者基类
  // TODO 把KD树和对应点的指针扔一个类里
  template <typename MP, typename K>
  std::tuple<std::vector<std::pair<MP, MP>>, Eigen::Vector4f, Eigen::Vector3f>
  Get(const std::vector<MP> &key, const K &key_kdt,
      const std::vector<MP> key_kdt_mps, std::vector<MP> query,
      const K &query_kdt, const std::vector<MP> query_kdt_mps,
      const std::unordered_map<size_t, std::unordered_map<uuids::uuid, MP>>
          &cls_index) const {
    // remove key in query
    std::unordered_map<uuids::uuid, MP> key_map;
    for (auto &item : key) {
      key_map[item->get_uuid()] = item;
    }
    for (auto itr = query.rbegin(); itr != query.rend(); ++itr) {
      if (key_map.find((*itr)->get_uuid()) != key_map.end()) {
        query.erase(std::next(itr).base());
      }
    }

    using MPKDADT = PointCloudAdaptor<std::vector<MP>>;
    using MPKDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, MPKDADT>, MPKDADT, 3>;

    auto key_neigh_dict =
        GetNeighbor(key, key_kdt, key_kdt_mps, neighbour_radius_);
    auto query_neigh_dict =
        GetNeighbor(query, query_kdt, query_kdt_mps, neighbour_radius_);
    auto [match, r_i, t_i] =
        CoarseReloc(key, query, key_neigh_dict, query_neigh_dict, cls_index);
    if (match.empty()) {
      return {match, r_i, t_i};
    }
    std::unordered_map<uuids::uuid, MP> mps_map_1;
    std::unordered_map<uuids::uuid, MP> mps_map_2;
    for (auto &item : match) {
      mps_map_1[item.first->get_uuid()] = item.first;
      mps_map_2[item.second->get_uuid()] = item.second;
      std::vector<MP> k1 = {item.first};
      std::vector<MP> k2 = {item.second};
      auto mps_neigh_1 =
          GetNeighbor(k1, key_kdt, key_kdt_mps, 2)[item.first->get_uuid()];
      auto mps_neigh_2 =
          GetNeighbor(k2, query_kdt, query_kdt_mps, 2)[item.second->get_uuid()];
      for (auto &item_ : mps_neigh_1) {
        mps_map_1[item_->get_uuid()] = item_;
      }
      for (auto &item_ : mps_neigh_2) {
        mps_map_2[item_->get_uuid()] = item_;
      }
    }
    std::vector<MP> mps_1;
    std::vector<MP> mps_2;
    mps_1.reserve(mps_map_1.size());
    mps_2.reserve(mps_map_2.size());
    for (auto &item : mps_map_1) {
      mps_1.push_back(item.second);
    }
    for (auto &item : mps_map_2) {
      mps_2.push_back(item.second);
    }
    auto [match_, r_, t_] = FineReloc(mps_1, mps_2, r_i, t_i);
    return {match_, r_, t_};
  }

  template <typename MP>
  std::tuple<std::vector<std::pair<MP, MP>>, Eigen::Vector4f, Eigen::Vector3f>
  CoarseReloc(
      std::vector<MP> key, const std::vector<MP> &query,
      const std::unordered_map<uuids::uuid, std::vector<MP>> key_neigh_dict,
      const std::unordered_map<uuids::uuid, std::vector<MP>> query_neigh_dict,
      const std::unordered_map<size_t, std::unordered_map<uuids::uuid, MP>>
          &cls_index) const {
    SortKey(key, key_neigh_dict, cls_index);

    std::unordered_map<uuids::uuid, std::pair<MP, MP>> mp_match;

    // 遍历所有查询点
    for (size_t i = 0; i < key.size(); ++i) {
      // 查找同类待匹配点
      if (cls_index.find(key[i]->get_cls()[0]) == cls_index.end()) {
        continue;
      }

      if (cls_index.find(key[i]->get_cls()[0])->second.size() == 1) {
        MP mp = cls_index.find(key[i]->get_cls()[0])->second.begin()->second;
        if (GetScaleScore(key[i]->get_scale(), mp->get_scale()) <
            scale_score_threshold_) {
          continue;
        }
        if (corr_distance(key[i]->get_descriptor(), mp->get_descriptor()) <
            desc_score_threshold_) {
          continue;
        }
        mp_match[mp->get_uuid()] = {mp, key[i]};
      } else {
        auto candidate_map = cls_index.find(key[i]->get_cls()[0])->second;

        std::vector<MP> candidate;
        candidate.reserve(candidate_map.size());
        for (auto &item : candidate_map) {
          candidate.push_back(item.second);
        }

        std::vector<float> scores(candidate.size(), 0);
        std::vector<std::pair<float, std::vector<std::pair<MP, MP>>>>
            score_match_pair(candidate.size(), {0, {}});
        //        std::vector<std::vector<std::pair<MP, MP>>> matchs;
        for (size_t j = 0; j < candidate.size(); ++j) {
          if (GetScaleScore(key[i]->get_scale(), candidate[j]->get_scale()) <
              scale_score_threshold_) {
            continue;
          }
          if (1 - corr_distance(key[i]->get_descriptor(),
                                candidate[j]->get_descriptor()) <
              desc_score_threshold_) {
            continue;
          }
          if (query_neigh_dict.find(candidate[j]->get_uuid()) !=
              query_neigh_dict.end()) {
            score_match_pair[j] = MatchByDesc(
                key[i], candidate[j],
                key_neigh_dict.find(key[i]->get_uuid())->second,
                query_neigh_dict.find(candidate[j]->get_uuid())->second);
          }
        }

        float max_score = 0;
        size_t max_match_size = 0;
        size_t max_idx = 0;
        for (size_t k = 0; k < score_match_pair.size(); ++k) {
          auto [s, m] = score_match_pair[k];
          if (s < pair_score_threshold_) {
            continue;
          }
          if (s > max_score) {
            max_score = s;
            max_idx = k;
          }
          // if (m.size() > max_match_size) {
          //   max_match_size = m.size();
          //   max_score = s;
          //   max_idx = k;
          // } else if (m.size() == max_match_size) {
          //   if (s > max_score) {
          //     max_score = s;
          //     max_idx = k;
          //   }
          // }
        }

        if (max_score == 0) {
          continue;
        }
        mp_match[candidate[max_idx]->get_uuid()] = {candidate[max_idx], key[i]};
      }

      // XXX  理论上需要4点？
      // 早停
      if (mp_match.size() > 2) {
        std::vector<MP> match_query;
        std::vector<MP> match_key;
        for (auto &item : mp_match) {
          match_query.push_back(item.second.first);
          match_key.push_back(item.second.second);
        }
        auto [best_match_key, best_match_query, r] =
            GetBestMatch(match_key, match_query);
        if (best_match_query.size() <= best_match_threshold_) {
          continue;
        }
        auto R = Eigen::Quaternionf(r).toRotationMatrix();
        auto t =
            best_match_query[0]->get_pos() - R * best_match_key[0]->get_pos();
        std::vector<std::pair<MP, MP>> ret_pair;
        for (size_t j = 0; j < best_match_key.size(); ++j) {
          ret_pair.push_back({best_match_key[j], best_match_query[j]});
        }
        return {ret_pair, r, t};
      }
    }
    return {{}, Eigen::Vector4f(0, 0, 0, 0), Eigen::Vector3f(0, 0, 0)};
  }

  template <typename MP>
  std::tuple<std::vector<std::pair<MP, MP>>, Eigen::Vector4f, Eigen::Vector3f>
  FineReloc(std::vector<MP> mps_1, std::vector<MP> mps_2, Eigen::Vector4f r_i,
            Eigen::Vector3f t_i) const {
    std::vector<Eigen::Vector3f> coor_1;
    std::vector<Eigen::Vector3f> coor_2;

    Eigen::Matrix3f R = Eigen::Quaternionf(r_i).toRotationMatrix();
    for (size_t i = 0; i < mps_1.size(); ++i) {
      coor_1.emplace_back(R * mps_1[i]->get_pos() + t_i);
    }
    for (size_t i = 0; i < mps_2.size(); ++i) {
      coor_2.push_back(mps_2[i]->get_pos());
    }

    std::vector<std::vector<float>> cost(coor_1.size(),
                                         std::vector<float>(coor_2.size(), 0));
    for (size_t i = 0; i < coor_1.size(); ++i) {
      for (size_t j = 0; j < coor_2.size(); ++j) {
        float desc_score = 1 - rgor::corr_distance(mps_1[i]->get_descriptor(),
                                                   mps_2[j]->get_descriptor());
        float dist = rgor::l2_distance(coor_1[i], coor_2[j]);
        // TODO add scale
        float dist_score = (1 - dist / fine_dist_threshold_) / 2;
        if (dist_score < 1e-3) {
          cost[i][j] = 1e-3;
        } else {
          cost[i][j] = 1e-3 + desc_score + dist_score;
        }
      }
    }

    auto match = Assignment(cost);

    std::vector<std::pair<MP, MP>> ret;
    ret.reserve(match.size());
    for (size_t i = 0; i < match.size(); ++i) {
      if (match[i].first >= mps_1.size() || match[i].second >= mps_2.size()) {
        continue;
      }
      if (cost[match[i].first][match[i].second] < 1e-2) {
        continue;
      }
      auto desc_score =
          1 - rgor::corr_distance(mps_1[match[i].first]->get_descriptor(),
                                  mps_2[match[i].second]->get_descriptor());
      if (desc_score < fine_desc_threshold_) {
        continue;
      }
      ret.push_back({mps_1[match[i].first], mps_2[match[i].second]});
    }

    std::vector<Eigen::Vector3f> src_mps;
    std::vector<Eigen::Vector3f> dst_mps;
    for (auto &item : ret) {
      src_mps.push_back(item.first->get_pos());
      dst_mps.push_back(item.second->get_pos());
    }
    auto [r_, t_] = OptimTransformation(src_mps, dst_mps, r_i, t_i);
    return {ret, r_, t_};
  }
  // TODO 检查cls_index线程安全，sub_map线程安全
private:
  template <typename MP>
  void SortKey(
      std::vector<MP> &key,
      const std::unordered_map<uuids::uuid, std::vector<MP>> &key_neigh_dict,
      const std::unordered_map<size_t, std::unordered_map<uuids::uuid, MP>>
          &cls_index) const {
    auto neigh_size_map = std::unordered_map<uuids::uuid, size_t>();
    for (size_t i = 0; i < key.size(); ++i) {
      neigh_size_map[key[i]->get_uuid()] =
          key_neigh_dict.find(key[i]->get_uuid())->second.size();
    }
    auto cmp = [&](MP a, MP b) {
      if (cls_index.find(a->get_cls()[0]) == cls_index.end()) {
        return false;
      }
      if (cls_index.find(b->get_cls()[0]) == cls_index.end()) {
        return true;
      }
      auto a_cls_size = cls_index.find(a->get_cls()[0])->second.size();
      auto b_cls_size = cls_index.find(b->get_cls()[0])->second.size();
      auto a_neigh_size = neigh_size_map[a->get_uuid()];
      auto b_neigh_size = neigh_size_map[b->get_uuid()];
      if (a_cls_size == b_cls_size) {
        return a_neigh_size > b_neigh_size;
      } else {
        return a_cls_size < b_cls_size;
      }
    };
    std::sort(key.begin(), key.end(), cmp);
  }

  template <typename MP, typename K>
  std::unordered_map<uuids::uuid, std::vector<MP>>
  GetNeighbor(const std::vector<MP> &key, const K &kdt,
              const std::vector<MP> kdt_mps, float neighbour_radius) const {
    if (neighbour_radius <= 0) {
      throw std::invalid_argument("neighbour_radius must be positive");
    }
    std::unordered_map<uuids::uuid, std::vector<MP>> ret;
    ret.reserve(key.size());
    for (size_t i = 0; i < key.size(); ++i) {
      if (key[i] == nullptr) {
        continue;
      }
      float query_pt[3] = {key[i]->get_pos()[0], key[i]->get_pos()[1],
                           key[i]->get_pos()[2]};
      std::vector<nanoflann::ResultItem<size_t, float>> indices_dists;
      nanoflann::RadiusResultSet<float, size_t> result_set(neighbour_radius,
                                                           indices_dists);

      kdt.findNeighbors(result_set, query_pt);
      ret[key[i]->get_uuid()] = {};
      for (auto &item : indices_dists) {
        ret[key[i]->get_uuid()].push_back(kdt_mps[item.first]);
      }
    }
    return ret;
  }

private:
  float neighbour_radius_;
  float scale_score_threshold_;
  float desc_score_threshold_;
  float pair_score_threshold_;
  float best_match_threshold_;
  float fine_dist_threshold_;
  float fine_desc_threshold_;
};
} // namespace rgor

#endif // RGOR_RELOCATION_H
