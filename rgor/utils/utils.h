/**
 * File: utils.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-05-26
 * Brief:
 */

#ifndef RGOR_UTILS_H
#define RGOR_UTILS_H

#include <Eigen/Core>
#include <algorithm>
#include <vector>

namespace rgor {

template <typename T>
float l2_distance(const T &v1, const T &v2) {
  return (v1 - v2).norm();
}

template <typename T>
float corr_distance(const T &v1, const T &v2) {
  auto ret = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-6);
  ret = ret < 0 ? 0 : ret;
  ret = ret > 1 ? 1 : ret;
  return 1 - ret;
}

template <typename T>
std::pair<T, T> EstimateScale(T w, T h, T depth, float fx, float fy) {
  T ws = depth * w / fx;
  T hs = depth * h / fy;
  if (ws < hs) {
    return {ws, hs};
  } else {
    return {hs, ws};
  }
}

// TODO 这个距离矩阵是不是能缓存，感觉很能优化一下
template <typename T>
T BestDescriptor(const std::vector<T> &candidates) {
  std::vector<std::vector<float>> dist(
      candidates.size(), std::vector<float>(candidates.size(), 0));
  for (size_t i = 0; i < candidates.size(); ++i) {
    for (size_t j = i; j < candidates.size(); ++j) {
      dist[i][j] = 1 - rgor::corr_distance(candidates[i], candidates[j]);
      dist[j][i] = dist[i][j];
    }
  }
  std::vector<float> scores;
  scores.reserve(candidates.size());
  for (auto &d : dist) {
    std::sort(d.begin(), d.end());
    scores.push_back(d[d.size() / 2]);
  }
  auto max_score = std::max_element(scores.begin(), scores.end());
  return candidates[std::distance(scores.begin(), max_score)];
}

template <typename T>
std::pair<T, T> BestScale(std::vector<std::pair<T, T>> scales) {
  std::vector<T> s1, s2;
  for (auto &s : scales) {
    s1.push_back(s.first);
    s2.push_back(s.second);
  }
  std::sort(s1.begin(), s1.end());

  std::sort(s2.begin(), s2.end());
  return {s1[s1.size() / 2], s2[s2.size() / 2]};
}

template <typename T>
std::pair<T, T> GetScaleMatchScore(std::pair<T, T> s1, std::pair<T, T> s2) {
  T score_s = s1.first > s2.first ? s2.first / s1.first : s1.first / s2.first;
  T score_l =
      s1.second > s2.second ? s2.second / s1.second : s1.second / s2.second;
  return {score_s, score_l};
}
template <typename T>
T GetScaleScore(std::pair<T, T> s1, std::pair<T, T> s2, bool max = true) {
  auto [ss, sl] = GetScaleMatchScore(s1, s2);
  auto ret = max ? std::max(ss, sl) : std::min(ss, sl);
  return ret;
}

template <typename T>
T MeanQuaternion(const std::vector<T> &quats) {
  if (quats.empty()) {
    return T(1, 0, 0,
             0);  // 返回单位四元数，如果四元数列表为空
  }

  T mean = quats[0];
  for (int i = 0; i < 3; ++i) {  // 进行几次迭代
    Eigen::Vector4f sum(0, 0, 0, 0);
    for (const auto &q : quats) {
      // 四元数乘法，计算四元数与均值的乘积
      T prod = mean.conjugate() * q;
      // 将结果累加到sum向量
      sum += prod.coeffs();
    }
    // 使用累加后的向量创建新的四元数，并更新均值
    mean.coeffs() = sum / quats.size();
    // 归一化
    mean.normalize();
  }
  return mean;
}

}  // namespace rgor

#endif  // RGOR_UTILS_H