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
#ifndef XI_LRU_CACHE_H
#define XI_LRU_CACHE_H

#include <chrono>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

template <typename Key, typename Value>
class LRUCache {
 private:
  struct Node {
    Key key;
    Value value;
    std::chrono::system_clock::time_point expire_time;
    Node* prev;
    Node* next;

    Node(const Key& k, const Value& v,
         const std::chrono::system_clock::time_point& exp)
        : key(k), value(v), expire_time(exp), prev(nullptr), next(nullptr) {}
  };

  Node* head;
  Node* tail;
  std::unordered_map<Key, Node*> cache_;
  size_t capacity_;
  std::function<void(const Key&, const Value&)> callback_;
  std::mutex mutex_;

  void add_to_head(Node* node) {
    node->prev = head;
    node->next = head->next;
    head->next->prev = node;
    head->next = node;
  }

  void remove_node(Node* node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }

  void move_to_head(Node* node) {
    remove_node(node);
    add_to_head(node);
  }

  void delete_node(const Key& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) return;

    Node* node = it->second;
    if (callback_) {
      callback_(node->key, node->value);
    }

    remove_node(node);
    cache_.erase(it);
    delete node;
  }

 public:
  LRUCache(size_t capacity,
           std::function<void(const Key&, const Value&)> callback = nullptr)
      : capacity_(capacity), callback_(callback) {
    head = new Node(Key(), Value(), std::chrono::system_clock::time_point());
    tail = new Node(Key(), Value(), std::chrono::system_clock::time_point());
    head->next = tail;
    tail->prev = head;
  }

  ~LRUCache() {
    Node* current = head->next;
    while (current != tail) {
      Node* next = current->next;
      delete current;
      current = next;
    }
    delete head;
    delete tail;
  }

  void put(const Key& key, const Value& value,
           std::chrono::seconds ttl = std::chrono::seconds::zero()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto expire_time = std::chrono::system_clock::now() + ttl;
      if (ttl == std::chrono::seconds::zero()) {
        expire_time = std::chrono::system_clock::time_point::max();
      }

      if (cache_.find(key) != cache_.end()) {
        Node* node = cache_[key];
        node->value = value;
        node->expire_time = expire_time;
        move_to_head(node);
      } else {
        Node* node = new Node(key, value, expire_time);
        cache_[key] = node;
        add_to_head(node);
      }
    }
    prune();
  }

  bool get(const Key& key, Value& value) {
    prune();

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return false;

    Node* node = it->second;
    if (node->expire_time <= std::chrono::system_clock::now()) {
      delete_node(key);
      return false;
    }

    move_to_head(node);
    value = node->value;
    return true;
  }

  void prune() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::system_clock::now();
    std::vector<Key> expired_keys;

    auto check_ptr = tail;
    while (check_ptr->expire_time <= now) {
      expired_keys.push_back(check_ptr->key);
      check_ptr = check_ptr->prev;
    }
    for (const auto& key : expired_keys) {
      delete_node(key);
    }

    while (cache_.size() > capacity_) {
      Node* lru_node = tail->prev;
      delete_node(lru_node->key);
    }
  }

  std::unordered_map<Key, Node*> get_cache() const { return cache_; }

  size_t size() const { return cache_.size(); }
};

#endif  // XI_LRU_CACHE_H