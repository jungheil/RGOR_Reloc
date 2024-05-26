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

#ifndef RGOR_VISUALIZATION_H_
#define RGOR_VISUALIZATION_H_

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "common/Frame.h"
#include "common/Map.h"

std::tuple<float, float, float> GetColor(const Eigen::VectorXf& desc) {
  // 求VectorXf最大值的idx
  size_t desc_size = desc.size();
  size_t max_idx = 0;
  float max_val = desc[0];
  for (int i = 1; i < desc.size(); i++) {
    if (desc[i] > max_val) {
      max_val = desc[i];
      max_idx = i;
    }
  }
  // hsv
  float h = max_idx * 360.0 / desc_size;
  float s = 1.0;
  float v = 1.0;
  // to rgb
  float c = v * s;
  float x = c * (1 - std::abs(fmod(h / 60.0, 2) - 1));
  float m = v - c;
  float r = 0, g = 0, b = 0;

  if (h >= 0 && h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (h >= 60 && h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (h >= 120 && h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (h >= 180 && h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (h >= 240 && h < 300) {
    r = x;
    g = 0;
    b = c;
  } else if (h >= 300 && h < 360) {
    r = c;
    g = 0;
    b = x;
  }

  return std::make_tuple(r, g, b);
}

struct GetMarkerArray {
  // TODO 输入mps应该const
  visualization_msgs::MarkerArray operator()(
      std::vector<rgor::MapPoint<rgor::KeyFrame>::Ptr>& mps, ros::Time time) {
    visualization_msgs::MarkerArray ret;
    for (size_t i = 0; i < mps.size(); i++) {
      if (mps[i] == nullptr) {
        continue;
      }

      visualization_msgs::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = time;
      marker.ns = "map_points";
      marker.id = i;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = i < last_idx ? visualization_msgs::Marker::MODIFY
                                   : visualization_msgs::Marker::ADD;
      marker.pose.position.x = mps[i]->get_pos()[0];
      marker.pose.position.y = mps[i]->get_pos()[1];
      marker.pose.position.z = mps[i]->get_pos()[2];
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = mps[i]->get_scale().first;
      marker.scale.y = mps[i]->get_scale().first;
      marker.scale.z = mps[i]->get_scale().first;
      marker.color.a = 0.6;
      auto [r, g, b] = GetColor(mps[i]->get_descriptor());
      marker.color.r = r;
      marker.color.g = g;
      marker.color.b = b;
      ret.markers.push_back(marker);
    }
    last_idx = mps.size();
    return ret;
  }

  size_t last_idx = 0;
};

#endif  // RGOR_VISUALIZATION_H_