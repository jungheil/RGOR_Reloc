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

#pragma once
#ifndef XI_FEATURE_DB_H
#define XI_FEATURE_DB_H

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_io.h"
#include "utils/LRUCache.h"
#include "uuid.h"

// IndexHNSWFlat
template <typename Quantizer>
class FaissIndex {
  // using IndexHNWSFlat = faiss::IndexHNSWFlat;
  // using IndexFlat = faiss::IndexFlat;

 public:
  // 针对 IndexFlat 类型的构造函数
  template <typename T = Quantizer>
  explicit FaissIndex(
      int dim,
      typename std::enable_if_t<std::is_same<T, faiss::IndexHNSWFlat>::value>* =
          nullptr)
      : dim_(dim) {
    try {
      quantizer_ = std::unique_ptr<Quantizer>(new Quantizer(dim, 16));
      index_ = std::unique_ptr<faiss::IndexIDMap2>(
          new faiss::IndexIDMap2(quantizer_.get()));
    } catch (const faiss::FaissException& e) {
      std::cerr << "Faiss error: " << e.what() << std::endl;
      throw;
    }
  }

  // 针对其他类型的构造函数
  template <typename T = Quantizer>
  explicit FaissIndex(
      int dim, typename std::enable_if_t<
                   !std::is_same<T, faiss::IndexHNSWFlat>::value>* = nullptr)
      : dim_(dim) {
    try {
      quantizer_ = std::unique_ptr<Quantizer>(new Quantizer(dim));
      index_ = std::unique_ptr<faiss::IndexIDMap2>(
          new faiss::IndexIDMap2(quantizer_.get()));
    } catch (const faiss::FaissException& e) {
      std::cerr << "Faiss error: " << e.what() << std::endl;
      throw;
    }
  }

  void add_feature(const uuids::uuid& uuid,
                   const Eigen::Ref<const Eigen::VectorXf>& feature) {
    std::unique_lock lock(mutex_);

    index_->add_with_ids(1, feature.data(), &idx_);
    id_to_uuid_[idx_] = uuid;
    idx_++;
  }

  std::vector<std::pair<uuids::uuid, float>> search_knn(
      const Eigen::Ref<const Eigen::VectorXf>& query, int k) {
    std::shared_lock lock(mutex_);
    std::vector<faiss::idx_t> ids(k);
    std::vector<float> distances(k);

    index_->search(1, query.data(), k, distances.data(), ids.data());

    std::vector<std::pair<uuids::uuid, float>> results;
    for (int i = 0; i < k; ++i) {
      if (ids[i] == -1) continue;
      auto uuid_iter = id_to_uuid_.find(ids[i]);
      if (uuid_iter == id_to_uuid_.end()) {
        continue;
      } else {
        results.emplace_back(uuid_iter->second, distances[i]);
      }
    }
    return results;
  }

  std::vector<std::pair<uuids::uuid, float>> search_range(
      const Eigen::Ref<const Eigen::VectorXf>& query, float radius) {
    std::shared_lock lock(mutex_);
    faiss::RangeSearchResult res(1);

    index_->range_search(1, query.data(), radius, &res);
    std::vector<std::pair<uuids::uuid, float>> results;
    for (int i = 0; i < res.lims[1]; ++i) {
      auto uuid_iter = id_to_uuid_.find(res.labels[i]);
      if (uuid_iter == id_to_uuid_.end()) {
        continue;
      } else {
        results.emplace_back(uuid_iter->second, res.distances[i]);
      }
    }
    return results;
  }

  Eigen::VectorXf reconstruct_feature(const uuids::uuid& uuid) {
    std::unique_lock lock(mutex_);
    auto it =
        std::find_if(id_to_uuid_.begin(), id_to_uuid_.end(),
                     [&uuid](const std::pair<faiss::idx_t, uuids::uuid>& p) {
                       return p.second == uuid;
                     });
    Eigen::VectorXf vec(dim_);
    if (it != id_to_uuid_.end()) {
      index_->reconstruct(it->first, vec.data());
    }
    return vec;
  }

  bool remove_feature(const uuids::uuid& uuid) {
    std::unique_lock lock(mutex_);
    auto it =
        std::find_if(id_to_uuid_.begin(), id_to_uuid_.end(),
                     [&uuid](const std::pair<faiss::idx_t, uuids::uuid>& p) {
                       return p.second == uuid;
                     });
    if (it == id_to_uuid_.end()) {
      return false;
    }
    try {
      faiss::IDSelectorBatch selector(1, &it->first);
      index_->remove_ids(selector);

    } catch (const faiss::FaissException& e) {
      std::cout << "Faiss error: " << e.what() << std::endl;
    }

    id_to_uuid_.erase(it);
    return true;
  }

  void save_index(std::string_view path) {
    std::shared_lock lock(mutex_);
    faiss::write_index(index_.get(), std::string(path).c_str());
  }

  void load_index(std::string_view path) {
    std::unique_lock lock(mutex_);
    index_ = std::unique_ptr<faiss::Index>(faiss::read_index(path.data()));
  }

 private:
  int dim_;
  std::unique_ptr<Quantizer> quantizer_;
  std::unique_ptr<faiss::Index> index_;

  faiss::idx_t idx_ = 0;
  std::unordered_map<faiss::idx_t, uuids::uuid> id_to_uuid_;

  std::shared_mutex mutex_;
};

class FeatureDB {
 public:
  explicit FeatureDB(int dim, size_t hot_capacity = 128,
                     size_t hot_expire_time = 10)
      : dim_(dim),
        hot_index_(dim),
        cold_index_(dim),
        hot_cache_(hot_capacity,
                   [this](const uuids::uuid& uuid, const void* ptr) {
                     auto feature = hot_index_.reconstruct_feature(uuid);
                     hot_index_.remove_feature(uuid);
                     cold_index_.add_feature(uuid, feature);
                   }),
        hot_capacity_(hot_capacity),
        hot_expire_time_(hot_expire_time) {}

  void add_feature(const uuids::uuid& uuid, const Eigen::VectorXf& feature) {
    std::unique_lock lock(mutex_);
    hot_index_.add_feature(uuid, feature);
    hot_cache_.put(uuid, nullptr, std::chrono::seconds(hot_expire_time_));
  }

  bool remove_feature(const uuids::uuid& uuid) {
    std::unique_lock lock(mutex_);
    if (hot_index_.remove_feature(uuid)) {
      return true;
    }
    // 如果热索引中没有,则尝试从冷索引中删除
    return cold_index_.remove_feature(uuid);
  }

  bool update_feature(const uuids::uuid& uuid, const Eigen::VectorXf& feature) {
    std::unique_lock lock(mutex_);

    void* ptr = nullptr;
    if (hot_cache_.get(uuid, ptr)) {
      hot_index_.remove_feature(uuid);
      hot_index_.add_feature(uuid, feature);
      hot_cache_.put(uuid, ptr, std::chrono::seconds(hot_expire_time_));
      return true;
    } else {
      if (cold_index_.remove_feature(uuid)) {
        hot_index_.add_feature(uuid, feature);
        hot_cache_.put(uuid, nullptr, std::chrono::seconds(hot_expire_time_));
        return true;
      } else {
        return false;
      }
    }
  }

  std::vector<std::pair<uuids::uuid, float>> search_knn(
      const Eigen::VectorXf& query, int k) {
    std::shared_lock lock(mutex_);

    // 分别在热索引和冷索引中搜索
    auto hot_results = hot_index_.search_knn(query, k);
    auto cold_results = cold_index_.search_knn(query, k);

    // 合并结果
    std::vector<std::pair<uuids::uuid, float>> merged_results;
    merged_results.reserve(hot_results.size() + cold_results.size());
    merged_results.insert(merged_results.end(), hot_results.begin(),
                          hot_results.end());
    merged_results.insert(merged_results.end(), cold_results.begin(),
                          cold_results.end());

    // 按距离排序
    std::sort(merged_results.begin(), merged_results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // 只保留前k个结果
    if (merged_results.size() > k) {
      merged_results.resize(k);
    }

    return merged_results;
  }

  std::vector<std::pair<uuids::uuid, float>> search_range(
      const Eigen::VectorXf& query, float radius) {
    std::shared_lock lock(mutex_);

    // 分别在热索引和冷索引中搜索
    auto hot_results = hot_index_.search_range(query, radius);
    auto cold_results = cold_index_.search_range(query, radius);

    // 合并结果
    std::vector<std::pair<uuids::uuid, float>> merged_results;
    merged_results.reserve(hot_results.size() + cold_results.size());
    merged_results.insert(merged_results.end(), hot_results.begin(),
                          hot_results.end());
    merged_results.insert(merged_results.end(), cold_results.begin(),
                          cold_results.end());

    // 按距离排序
    std::sort(merged_results.begin(), merged_results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    return merged_results;
  }

  void save_index(std::string_view path) {
    std::shared_lock lock(mutex_);
    std::string hot_path = std::string(path) + ".hot";
    std::string cold_path = std::string(path) + ".cold";
    hot_index_.save_index(hot_path);
    cold_index_.save_index(cold_path);
  }

  void load_index(std::string_view path) {
    std::unique_lock lock(mutex_);
    std::string hot_path = std::string(path) + ".hot";
    std::string cold_path = std::string(path) + ".cold";
    hot_index_.load_index(hot_path);
    cold_index_.load_index(cold_path);
  }

 private:
  int dim_;
  size_t hot_capacity_;
  size_t hot_expire_time_;

  // 热索引使用IndexFlat
  FaissIndex<faiss::IndexFlat> hot_index_;
  // 冷索引使用IndexHNSWFlat
  FaissIndex<faiss::IndexHNSWFlat> cold_index_;

  LRUCache<uuids::uuid, void*> hot_cache_;

  std::shared_mutex mutex_;
};

#endif  // XI_FEATURE_DB_H