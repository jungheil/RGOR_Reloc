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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Optimizer.h"
// #include "common/Frame.h"
// #include "common/Map.h"
#include "common/NeoMap.h"
#include "common/Params.h"
#include "utils/FeatureDB.h"
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
std::tuple<std::vector<MP>, std::vector<MP>, Eigen::Vector3f, Eigen::Vector4f>
GetBestMatch(const std::vector<MP> &mps_1, const std::vector<MP> &mps_2) {
  assert(mps_1.size() == mps_2.size());
  size_t points_size = mps_1.size();
  std::vector<size_t> ret_idx(points_size);
  for (size_t i = 0; i < points_size; ++i) {
    ret_idx[i] = i;
  }
  std::vector<std::vector<float>> dist_matrix(
      points_size, std::vector<float>(points_size, 0));
  // std::vector<std::vector<Eigen::Quaternionf>> q_matrix(
  //     points_size, std::vector<Eigen::Quaternionf>(points_size));
  for (size_t i = 0; i < points_size; ++i) {
    for (size_t j = i + 1; j < points_size; ++j) {
      float dist_1 = (mps_1[i]->get_pos() - mps_1[j]->get_pos()).norm();
      float dist_2 = (mps_2[i]->get_pos() - mps_2[j]->get_pos()).norm();
      dist_matrix[i][j] = std::abs(dist_1 - dist_2);
      dist_matrix[j][i] = std::abs(dist_1 - dist_2);
      // Eigen::Quaternionf q;
      // q.setFromTwoVectors(mps_1[j]->get_pos() - mps_1[i]->get_pos(),
      //                     mps_2[j]->get_pos() - mps_2[i]->get_pos());
      // q_matrix[i][j] = q;
      // // 这里应该不需要inv（好像需要欸）
      // q_matrix[j][i] = q.inverse();
    }
  }

  // 根据距离排除离群点
  while (points_size > 2) {
    // Calculate mean distance for each point
    std::vector<float> dist_mean(points_size);
    for (size_t i = 0; i < points_size; ++i) {
      float sum = 0;
      for (size_t j = 0; j < points_size; ++j) {
        sum += dist_matrix[i][j];
      }
      dist_mean[i] = sum / points_size;
    }

    // Find max mean
    size_t max_idx = 0;
    float max_dist = dist_mean[0];
    for (size_t i = 1; i < points_size; ++i) {
      if (dist_mean[i] > max_dist) {
        max_dist = dist_mean[i];
        max_idx = i;
      }
    }

    // Calculate mean and standard deviation of all means
    float total_mean = 0;
    for (const float &mean : dist_mean) {
      total_mean += mean;
    }
    total_mean -= max_dist;
    total_mean /= points_size - 1;

    float variance = 0;
    for (const float &mean : dist_mean) {
      variance += (mean - total_mean) * (mean - total_mean);
    }
    variance -= (max_dist - total_mean) * (max_dist - total_mean);
    variance /= (points_size - 1);
    float sigma = std::sqrt(variance);

    // Check if max is an outlier (> 3 sigma from mean)
    if (max_dist > total_mean + 3 * sigma) {
      // Remove the outlier point
      dist_matrix.erase(dist_matrix.begin() + max_idx);
      for (auto &row : dist_matrix) {
        row.erase(row.begin() + max_idx);
      }
      // q_matrix.erase(q_matrix.begin() + max_idx);
      // for (auto &row : q_matrix) {
      //   row.erase(row.begin() + max_idx);
      // }
      ret_idx.erase(ret_idx.begin() + max_idx);
      points_size--;
    } else {
      break;  // No more outliers found
    }
  }

  // 假设所有飞机都是垂直于地面起飞的，只有绕垂直于地平面的轴旋转，映射到地平面上，根据旋转排除离群点(应该是xoz平面)
  std::vector<std::vector<float>> angle_matrix(points_size,
                                               std::vector<float>(points_size));

  for (size_t i = 0; i < points_size; ++i) {
    for (size_t j = i + 1; j < points_size; ++j) {
      // 获取第一组点对在xoz平面上的向量
      Eigen::Vector2f vec1_1(mps_1[ret_idx[i]]->get_pos()[0],
                             mps_1[ret_idx[i]]->get_pos()[2]);
      Eigen::Vector2f vec1_2(mps_1[ret_idx[j]]->get_pos()[0],
                             mps_1[ret_idx[j]]->get_pos()[2]);
      Eigen::Vector2f vec1 = vec1_2 - vec1_1;

      // 获取第二组点对在xoz平面上的向量
      Eigen::Vector2f vec2_1(mps_2[ret_idx[i]]->get_pos()[0],
                             mps_2[ret_idx[i]]->get_pos()[2]);
      Eigen::Vector2f vec2_2(mps_2[ret_idx[j]]->get_pos()[0],
                             mps_2[ret_idx[j]]->get_pos()[2]);
      Eigen::Vector2f vec2 = vec2_2 - vec2_1;

      // 计算向量夹角（弧度）
      float angle1 = std::atan2(vec1[1], vec1[0]);
      float angle2 = std::atan2(vec2[1], vec2[0]);

      // 计算角度差的绝对值（归一化到 [-π, π]）
      float angle_diff = angle2 - angle1;
      if (angle_diff > M_PI) {
        angle_diff = 2 * M_PI - angle_diff;
      } else if (angle_diff < -M_PI) {
        angle_diff = 2 * M_PI + angle_diff;
      }

      // 存储到对称矩阵
      angle_matrix[i][j] = angle_diff;
      angle_matrix[j][i] = -angle_diff;
    }
  }

  // 迭代去除离群点
  while (points_size > 2) {
    // Calculate mean angle for each point
    std::vector<float> angle_means(points_size);
    for (size_t i = 0; i < points_size; ++i) {
      float sum = 0;
      for (size_t j = 0; j < points_size; ++j) {
        sum += angle_matrix[i][j];
      }
      angle_means[i] = sum / points_size;
    }

    float angle_mean = 0;
    for (const float &mean : angle_means) {
      angle_mean += mean;
    }
    angle_mean /= points_size;

    // Find max mean
    size_t max_idx = 0;
    float max_error = angle_means[0] - angle_mean;
    for (size_t i = 1; i < points_size; ++i) {
      if (angle_means[i] - angle_mean > max_error) {
        max_error = angle_means[i] - angle_mean;
        max_idx = i;
      }
    }

    // Calculate mean and standard deviation of all means
    float total_mean = 0;
    for (const float &mean : angle_means) {
      total_mean += mean;
    }
    total_mean -= angle_means[max_idx];
    total_mean /= points_size - 1;

    float variance = 0;
    for (const float &mean : angle_means) {
      variance += (mean - total_mean) * (mean - total_mean);
    }
    variance -= (angle_means[max_idx] - total_mean) *
                (angle_means[max_idx] - total_mean);
    variance /= (points_size - 1);
    float sigma = std::sqrt(variance);

    // Check if max is an outlier (> 3 sigma from mean)
    if (angle_means[max_idx] > total_mean + 3 * sigma) {
      // Remove the outlier point
      angle_matrix.erase(angle_matrix.begin() + max_idx);
      for (auto &row : angle_matrix) {
        row.erase(row.begin() + max_idx);
      }
      dist_matrix.erase(dist_matrix.begin() + max_idx);
      for (auto &row : dist_matrix) {
        row.erase(row.begin() + max_idx);
      }
      ret_idx.erase(ret_idx.begin() + max_idx);
      points_size--;
    } else {
      break;
    }
  }

  //   // 根据旋转排除离群点
  //   // 根据dist_matrix中误差最小的作为原点
  //   float min_dist = std::numeric_limits<float>::max();
  //   size_t origin_idx = 0;
  //   size_t origin_ret_idx = 0;
  //   for (size_t i = 0; i < points_size; ++i) {
  //       float dist = 0;
  //       for (size_t j = 0; j < points_size; ++j) {
  //           float dist = dist_matrix[i][j];
  //           if (dist < min_dist) {
  //               min_dist = dist;
  //               origin_idx = i;
  //           }
  //       }
  //   }
  //   origin_ret_idx = ret_idx[origin_idx];
  //   ret_idx.erase(ret_idx.begin() + origin_idx);

  //   // 基于原点 origin
  //   std::vector<std::vector<float>> q_diff_matrix(
  //           points_size, std::vector<float>(points_size, 0));
  //   for (size_t i = 0; i < points_size; ++i) {
  //       for (size_t j = i + 1; j < points_size; ++j) {
  //           q_diff_matrix[i][j] = (q_matrix[origin_idx][i] *
  //                                  q_matrix[origin_idx][j].inverse())
  //                   .angularDistance(Eigen::Quaternionf::Identity());
  //           q_diff_matrix[j][i] = q_diff_matrix[i][j];
  //       }
  //   }

  //   q_diff_matrix.erase(q_diff_matrix.begin() + origin_idx);
  //   for (auto &row : q_diff_matrix) {
  //       row.erase(row.begin() + origin_idx);
  //   }
  //   points_size--;

  //   // 迭代去除离群点
  // while (points_size > 2) {
  //   // Calculate mean distance for each point
  //   std::vector<float> q_diff_mean(points_size);
  //   for (size_t i = 0; i < points_size; ++i) {mps_1[origin_idx].position;
  //     float sum = 0;
  //     for (size_t j = 0; j < points_size; ++j) {
  //       sum += q_diff_matrix[i][j];
  //     }
  //     q_diff_mean[i] = sum / points_size;
  //   }

  //   // Find max mean
  //   size_t max_idx = 0;
  //   float max_dist = q_diff_mean[0];
  //   for (size_t i = 1; i < points_size; ++i) {
  //     if (q_diff_mean[i] > max_dist) {
  //       max_dist = q_diff_mean[i];
  //       max_idx = i;
  //     }
  //   }

  //   // Calculate mean and standard deviation of all means
  //   float total_mean = 0;
  //   for (const float &mean : q_diff_mean) {
  //     total_mean += mean;
  //   }
  //   total_mean -= max_dist;
  //   total_mean /= points_size - 1;

  //   float variance = 0;
  //   for (const float &mean : q_diff_mean) {
  //     variance += (mean - total_mean) * (mean - total_mean);
  //   }
  //   variance -= (max_dist - total_mean) * (max_dist - total_mean);
  //   variance /= (points_size - 1);
  //   float sigma = std::sqrt(variance);

  //   // Check if max is an outlier (> 3 sigma from mean)
  //   if (max_dist > total_mean + 3 * sigma) {
  //     // Remove the outlier point
  //     q_diff_matrix.erase(q_diff_matrix.begin() + max_idx);
  //     for (auto &row : q_diff_matrix) {
  //       row.erase(row.begin() + max_idx);
  //     }
  //     dist_matrix.erase(dist_matrix.begin() + max_idx);
  //     for (auto &row : dist_matrix) {
  //       row.erase(row.begin() + max_idx);
  //     }
  //     q_matrix.erase(q_matrix.begin() + max_idx);
  //     for (auto &row : q_matrix) {
  //       row.erase(row.begin() + max_idx);
  //     }
  //     ret_idx.erase(ret_idx.begin() + max_idx);
  //     points_size--;
  //   } else {
  //     break;  // No more outliers found
  //   }
  // }

  if (points_size < 3) {
    return {{}, {}, {}, {}};
  }

  //     float total_err = std::numeric_limits<float>::max();
  //     size_t total_min_idx = 0;
  //     for (size_t i = 0; i < points_size; ++i) {
  //       float err = 0;
  //       for (size_t j = 0; j < points_size; ++j) {
  //         err += dist_matrix[i][j];
  //       }
  //       if (err < total_err) {
  //         total_err = err;
  //         total_min_idx = i;
  //       }
  //     }

  float min_dist = std::numeric_limits<float>::max();
  size_t origin_idx = 0;
  for (size_t i = 0; i < points_size; ++i) {
    float dist = 0;
    for (size_t j = 0; j < points_size; ++j) {
      float dist = dist_matrix[i][j];
      if (dist < min_dist) {
        min_dist = dist;
        origin_idx = i;
      }
    }
  }

  std::vector<MP> ret_1;
  std::vector<MP> ret_2;
  for (size_t i = 0; i < points_size; ++i) {
    ret_1.push_back(mps_1[ret_idx[i]]);
    ret_2.push_back(mps_2[ret_idx[i]]);
  }

  float angle = 0;
  for (size_t i = 0; i < points_size; ++i) {
    angle += angle_matrix[i][origin_idx];
  }
  angle /= points_size;

  // TODO 请判断这个angle是正还是负
  Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
  q = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitY()) * q;

  Eigen::Vector3f t =
      mps_2[origin_idx]->get_pos() - q * mps_1[origin_idx]->get_pos();

  return {ret_1, ret_2, t, q.coeffs()};

  // size_t min_idx_1 = 0;
  // size_t min_idx_2 = 0;
  // while (points_size > 1) {
  //   std::vector<float> dist_mean(points_size);
  //   for (size_t i = 0; i < points_size; ++i) {
  //     float sum = 0;
  //     for (size_t j = 0; j < points_size; ++j) {
  //       sum += dist_matrix[i][j];
  //     }
  //     dist_mean[i] = sum / points_size;
  //   }

  //   size_t max_idx = 0;
  //   float max_dist = 0;
  //   float min_dist_1 = std::numeric_limits<float>::max();
  //   float min_dist_2 = std::numeric_limits<float>::max();
  //   for (size_t i = 0; i < points_size; ++i) {
  //     if (dist_mean[i] > max_dist) {
  //       max_dist = dist_mean[i];
  //       max_idx = i;
  //     }
  //     if (dist_mean[i] < min_dist_2) {
  //       min_dist_2 = dist_mean[i];
  //       min_idx_2 = i;
  //       if (min_dist_2 < min_dist_1) {
  //         size_t t_idx = min_idx_1;
  //         float t_dist = min_dist_1;
  //         min_idx_1 = min_idx_2;
  //         min_idx_2 = t_idx;
  //         min_dist_1 = min_dist_2;
  //         min_dist_2 = t_dist;
  //       }
  //     }
  //   }
  //   if (max_dist > 0.5) {
  //     dist_matrix.erase(dist_matrix.begin() + max_idx);
  //     for (auto itr = dist_matrix.begin(); itr != dist_matrix.end(); ++itr) {
  //       itr->erase(itr->begin() + max_idx);
  //     }
  //     q_matrix.erase(q_matrix.begin() + max_idx);
  //     for (auto itr = q_matrix.begin(); itr != q_matrix.end(); ++itr) {
  //       itr->erase(itr->begin() + max_idx);
  //     }
  //     ret_idx.erase(ret_idx.begin() + max_idx);
  //     points_size--;
  //   } else {
  //     break;
  //   }
  // }

  // // TODO 过滤异常点

  // if (points_size < 2) {
  //   return {{}, {}, {}};
  // }

  // float total_err = std::numeric_limits<float>::max();
  // size_t total_min_idx = 0;
  // for (size_t i = 0; i < points_size; ++i) {
  //   float err = 0;
  //   for (size_t j = 0; j < points_size; ++j) {
  //     err += dist_matrix[i][j];
  //   }
  //   if (err < total_err) {
  //     total_err = err;
  //     total_min_idx = i;
  //   }
  // }

  // std::vector<MP> ret_1;
  // std::vector<MP> ret_2;
  // for (size_t i = 0; i < points_size; ++i) {
  //   ret_1.push_back(mps_1[ret_idx[i]]);
  //   ret_2.push_back(mps_2[ret_idx[i]]);
  // }
  // // min_idx_1 换到0
  // std::swap(ret_1[0], ret_1[total_min_idx]);
  // std::swap(ret_2[0], ret_2[total_min_idx]);
  // std::swap(ret_idx[0], ret_idx[total_min_idx]);

  // std::vector<Eigen::Vector3f> ret_1_pos;
  // std::vector<Eigen::Vector3f> ret_2_pos;
  // for (size_t i = 0; i < points_size; ++i) {
  //   ret_1_pos.push_back(ret_1[i]->get_pos());
  //   ret_2_pos.push_back(ret_2[i]->get_pos());
  // }

  // // TODO 获取平移旋转初值

  // return {ret_1, ret_2, {0, 0, 0, 1}};
}

template <typename MP>
std::pair<float, std::vector<std::pair<MP, MP>>> MatchByDesc(
    MP key, MP query, const std::vector<MP> &key_neigh,
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
      // TODO ARG
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
  template <typename MAP, typename MP>
  std::tuple<std::vector<std::pair<uuids::uuid, uuids::uuid>>, Eigen::Vector4f,
             Eigen::Vector3f>
  Get(const std::shared_ptr<MAP> query_map,
      const std::shared_ptr<MAP> candidate_map,
      const std::optional<std::vector<uuids::uuid>> &query_selector =
          std::nullopt,
      const std::optional<std::vector<uuids::uuid>> &candidate_selector =
          std::nullopt) const {
    std::vector<uuids::uuid> query_uuid = query_selector.has_value()
                                              ? query_selector.value()
                                              : query_map->GetMPSUUID();

    std::vector<uuids::uuid> candidate_uuid = candidate_selector.has_value()
                                                  ? candidate_selector.value()
                                                  : candidate_map->GetMPSUUID();

    // remove candidate in query
    candidate_uuid.erase(
        std::remove_if(candidate_uuid.begin(), candidate_uuid.end(),
                       [&query_uuid](const auto &element) {
                         return std::find(query_uuid.begin(), query_uuid.end(),
                                          element) != query_uuid.end();
                       }),
        candidate_uuid.end());

//    using MPKDADT = PointCloudAdaptor<std::vector<MP>>;
//    using MPKDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
//        nanoflann::L2_Simple_Adaptor<float, MPKDADT>, MPKDADT, 3>;

    auto [match, r_i, t_i] = CoarseReloc<MAP, MP>(query_uuid, candidate_uuid,
                                                  query_map, candidate_map);
    if (match.empty()) {
      return {match, r_i, t_i};
    }
    std::unordered_map<uuids::uuid, MP> mps_map_q;
    std::unordered_map<uuids::uuid, MP> mps_map_c;
    for (auto &item : match) {
      mps_map_q[item.first] = query_map->GetMPByUUID(item.first);
      mps_map_c[item.second] = candidate_map->GetMPByUUID(item.second);
      auto query_neigh = query_map->GetNeighbors(item.first, neighbour_radius_);
      auto candidate_neigh =
          candidate_map->GetNeighbors(item.second, neighbour_radius_);
      //      auto mps_neigh_1 =
      //          GetNeighbor(k1, key_kdt, key_kdt_mps,
      //          2)[item.first->get_uuid()];
      //      auto mps_neigh_2 =
      //          GetNeighbor(k2, query_kdt, query_kdt_mps,
      //          2)[item.second->get_uuid()];
      for (const auto &uuid : query_neigh) {
        mps_map_q[uuid] = query_map->GetMPByUUID(uuid);
      }
      for (const auto &uuid : candidate_neigh) {
        mps_map_c[uuid] = candidate_map->GetMPByUUID(uuid);
      }
    }
    std::vector<MP> mps_q;
    std::vector<MP> mps_c;
    mps_q.reserve(mps_map_q.size());
    mps_c.reserve(mps_map_c.size());
    for (auto &item : mps_map_q) {
      mps_q.push_back(item.second);
    }
    for (auto &item : mps_map_c) {
      mps_c.push_back(item.second);
    }
    auto [match_, r_, t_] = FineReloc(mps_q, mps_c, r_i, t_i);
    return {match_, r_, t_};
  }

  template <typename MAP, typename MP>
  std::tuple<std::vector<std::pair<uuids::uuid, uuids::uuid>>, Eigen::Vector4f,
             Eigen::Vector3f>
  CoarseReloc(const std::vector<uuids::uuid> query_uuid,
              const std::vector<uuids::uuid> candidate_uuid,
              const std::shared_ptr<MAP> query_map,
              const std::shared_ptr<MAP> candidate_map) const {
    // SortQuery 在feature_index中已经没有意义，index花销太大
    // std::vector<MP> query_uuid_vec;
    // query_uuid_vec.reserve(query_uuid.size());
    // for (auto &item : query_uuid) {
    //   query_uuid_vec.push_back(query_map->GetMP(item));
    // }
    // SortKey(key, key_neigh_dict, feature_index);

    std::unordered_map<uuids::uuid, uuids::uuid> mp_match;

    std::vector<std::pair<uuids::uuid, uuids::uuid>> ret_pair;
    Eigen::Vector3f ret_t;
    Eigen::Vector4f ret_r;

    if (query_uuid.empty() || candidate_uuid.empty()) {
      return {ret_pair, ret_r, ret_t};
    }

    // 遍历所有查询点
    for (size_t i = 0; i < query_uuid.size(); ++i) {
      // 查找同类待匹配点

      MP query_mp;
      try {
        query_mp = query_map->GetMPByUUID(query_uuid[i]);
      } catch (NeoMap::DataNotFoundException) {
        continue;
      }

      auto sim_mps =
          candidate_map->GetSimilar(query_mp->get_descriptor(), SEARCH_RADIUS);
      if (sim_mps.empty()) {
        continue;
      }

      if (sim_mps.size() == 1) {
        auto mp = candidate_map->GetMPByUUID(sim_mps[0]);

        if (GetScaleScore(query_mp->get_scale(), mp->get_scale()) <
            scale_score_threshold_) {
          continue;
        }
        if (corr_distance(query_mp->get_descriptor(), mp->get_descriptor()) <
            desc_score_threshold_) {
          continue;
        }
        mp_match[query_uuid[i]] = {sim_mps[0]};
      } else {
        std::vector<float> scores(sim_mps.size(), 0);
        // TODO MP 改成UUID
        std::vector<std::pair<float, std::vector<std::pair<MP, MP>>>>
            score_match_pair(sim_mps.size(), {0, {}});
        //        std::vector<std::vector<std::pair<MP, MP>>> matchs;
        for (size_t j = 0; j < sim_mps.size(); ++j) {
          auto mp = candidate_map->GetMPByUUID(sim_mps[j]);
          if (GetScaleScore(query_mp->get_scale(), mp->get_scale()) <
              scale_score_threshold_) {
            continue;
          }
          if (1 - corr_distance(query_mp->get_descriptor(),
                                mp->get_descriptor()) <
              desc_score_threshold_) {
            continue;
          }
          try {
          } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
          }
          auto query_neigh_uuid =
              query_map->GetNeighbors(query_uuid[i], neighbour_radius_);
          auto candidate_neigh_uuid =
              candidate_map->GetNeighbors(sim_mps[j], neighbour_radius_);
          if (query_neigh_uuid.empty() || candidate_neigh_uuid.empty()) {
            continue;
          }
          std::vector<MP> query_neigh;
          query_neigh.reserve(query_neigh_uuid.size());
          std::vector<MP> candidate_neigh;
          candidate_neigh.reserve(candidate_neigh_uuid.size());
          for (auto &item : query_neigh_uuid) {
            query_neigh.push_back(query_map->GetMPByUUID(item));
          }
          for (auto &item : candidate_neigh_uuid) {
            candidate_neigh.push_back(candidate_map->GetMPByUUID(item));
          }
          score_match_pair[j] =
              MatchByDesc(query_mp, mp, query_neigh, candidate_neigh);
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
          //           }
        }

        if (max_score == 0) {
          continue;
        }
        mp_match[candidate_uuid[max_idx]] = query_uuid[i];
      }

      if (mp_match.size() > 2) {
        std::vector<MP> match_query;
        std::vector<MP> match_candidate;
        for (auto [c, q] : mp_match) {
          match_query.push_back(query_map->GetMPByUUID(q));
          match_candidate.push_back(candidate_map->GetMPByUUID(c));
        }
        auto [best_match_query, best_match_candidate, t, r] =
            GetBestMatch(match_query, match_candidate);

        ret_pair.clear();
        for (size_t j = 0; j < best_match_query.size(); ++j) {
          ret_pair.push_back({best_match_query[j]->get_uuid(),
                              best_match_candidate[j]->get_uuid()});
        }
        ret_t = t;
        ret_r = r;
        // 早停
        if (best_match_query.size() > best_match_threshold_) {
          return {ret_pair, ret_r, ret_t};
        }
      }
    }
    return {ret_pair, ret_r, ret_t};
  }

  template <typename MP>
  std::tuple<std::vector<std::pair<uuids::uuid, uuids::uuid>>, Eigen::Vector4f,
             Eigen::Vector3f>
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

    std::vector<std::pair<uuids::uuid, uuids::uuid>> ret_;
    for (auto &item : ret) {
      ret_.push_back({item.first->get_uuid(), item.second->get_uuid()});
    }
    return {ret_, r_, t_};
  }
  // TODO 检查cls_index线程安全，sub_map线程安全
 private:
  template <typename MP>
  void SortKey(
      std::vector<MP> &key,
      const std::unordered_map<uuids::uuid, std::vector<MP>> &key_neigh_dict,
      const FeatureDB &feature_index) const {
    auto neigh_size_map = std::unordered_map<uuids::uuid, size_t>();
    for (size_t i = 0; i < key.size(); ++i) {
      neigh_size_map[key[i]->get_uuid()] =
          key_neigh_dict.find(key[i]->get_uuid())->second.size();
    }
    auto cmp = [&](MP a, MP b) {
      auto a_sim_size =
          feature_index.search_range(a->get_descriptor(), SEARCH_RADIUS).size();
      auto b_sim_size =
          feature_index.search_range(b->get_descriptor(), SEARCH_RADIUS).size();

      auto a_neigh_size = neigh_size_map[a->get_uuid()];
      auto b_neigh_size = neigh_size_map[b->get_uuid()];
      if (a_sim_size == b_sim_size) {
        return a_neigh_size > b_neigh_size;
      } else {
        return a_sim_size < b_sim_size;
      }
    };
    std::sort(key.begin(), key.end(), cmp);
  }

  template <typename MP, typename K>
  std::unordered_map<uuids::uuid, std::vector<MP>> GetNeighbor(
      const std::vector<MP> &key, const K &kdt, const std::vector<MP> kdt_mps,
      float neighbour_radius) const {
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
}  // namespace rgor

#endif  // RGOR_RELOCATION_H
