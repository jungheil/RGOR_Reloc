/**
 * File: KMAssignment.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-05-26
 * Brief:
 */

#pragma once
#ifndef XI_KMASSIGNMENT_H
#define XI_KMASSIGNMENT_H

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <limits>
#include <vector>

namespace KM {

template <typename T>
bool FindPath_(size_t x, const std::vector<std::vector<T>> &cost_matrix,
               size_t ny_, std::vector<T> &lx_, std::vector<T> &ly_,
               std::vector<size_t> &match_, std::vector<T> &slack_,
               std::vector<bool> &visx_, std::vector<bool> &visy_) {
  T temp_delta;
  visx_[x] = true;
  for (size_t y = 0; y < ny_; ++y) {
    if (visy_[y]) continue;
    temp_delta = lx_[x] + ly_[y] - cost_matrix[x][y];
    if (abs(temp_delta) <= std::numeric_limits<T>::epsilon() * 1e3) {
      visy_[y] = true;
      if (match_[y] == -1 || FindPath_(match_[y], cost_matrix, ny_, lx_, ly_,
                                       match_, slack_, visx_, visy_)) {
        match_[y] = x;
        return true;
      }
    } else if (slack_[y] > temp_delta)
      slack_[y] = temp_delta;
  }
  return false;
}

template <typename T>
std::vector<std::pair<size_t, size_t>> KM_(
    const std::vector<std::vector<T>> &cost_matrix) {
  size_t nx_ = cost_matrix.size();
  size_t ny_ = cost_matrix[0].size();

  auto lx_ = std::vector<T>(nx_, 0);
  auto ly_ = std::vector<T>(ny_, 0);
  auto match_ = std::vector<size_t>(ny_, -1);
  auto slack_ = std::vector<T>(ny_, 0);
  auto visx_ = std::vector<bool>(nx_, false);
  auto visy_ = std::vector<bool>(ny_, false);

  for (size_t x = 0; x < nx_; ++x) {
    slack_ = std::vector<T>(ny_, std::numeric_limits<T>::max());
    while (true) {
      visx_.assign(nx_, false);
      visy_.assign(ny_, false);
      if (FindPath_<T>(x, cost_matrix, ny_, lx_, ly_, match_, slack_, visx_,
                       visy_))
        break;
      else {
        T delta = std::numeric_limits<T>::max();
        for (size_t i = 0; i < ny_; ++i)
          if (!visy_[i] && delta > slack_[i]) delta = slack_[i];
        for (size_t i = 0; i < nx_; ++i)
          if (visx_[i]) lx_[i] -= delta;
        for (size_t j = 0; j < ny_; ++j) {
          if (visy_[j])
            ly_[j] += delta;
          else
            slack_[j] -= delta;
        }
      }
    }
  }

  std::vector<std::pair<size_t, size_t>> ret;
  for (size_t i = 0; i < ny_; ++i) {
    if (match_[i] != -1) {
      ret.emplace_back(match_[i], i);
    }
  }
  return ret;
}

}  // namespace KM

template <typename T>
std::vector<std::pair<size_t, size_t>> Assignment(
    const std::vector<std::vector<T>> &cost_matrix) {
  if (cost_matrix.empty()) {
    return {};
  } else if (cost_matrix[0].empty()) {
    return {};
  }
  T max_value = std::numeric_limits<T>::min();
  for (const auto &sv : cost_matrix) {
    auto sub_max = *std::max_element(sv.begin(), sv.end());
    max_value = sub_max > max_value ? sub_max : max_value;
  }
  T min_value = std::numeric_limits<T>::max();
  for (const auto &sv : cost_matrix) {
    auto sub_min = *std::min_element(sv.begin(), sv.end());
    min_value = sub_min < min_value ? sub_min : min_value;
  }
  assert(!(min_value < 0 && max_value > 0));
  // TODO 不能等于0
  if (cost_matrix.size() <= cost_matrix[0].size()) {
    return KM::KM_(cost_matrix);
  } else {
    std::vector<std::vector<T>> cost_matrix_t(
        cost_matrix[0].size(), std::vector<T>(cost_matrix.size(), 0));
    for (size_t i = 0; i < cost_matrix.size(); ++i)
      for (size_t j = 0; j < cost_matrix[0].size(); ++j)
        cost_matrix_t[j][i] = cost_matrix[i][j];
    auto ret = KM::KM_(cost_matrix_t);
    std::vector<std::pair<size_t, size_t>> ret_;
    for (auto i : ret) {
      ret_.emplace_back(i.second, i.first);
    }
    return ret_;
  }
}

#endif  // XI_KMASSIGNMENT_H
