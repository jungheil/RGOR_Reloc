/**
 * Copyright (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * RGOR (Relocalization with Generalized Object Recognition) is licensed
 * under Mulan PSL v2.
 */

#include <ros/ros.h>
#include <memory>
#include <vector>
#include <string>

#include "../rgor/include/Global.h"
#include "neo_map_converter.h"

namespace rgor {

class MultiMapNode {
 public:
  MultiMapNode(ros::NodeHandle& nh, int num_maps = 3) 
    : num_maps_(num_maps) {
    // 初始化全局地图
    gb_map_ = std::make_shared<GBMap>();
    
    // 创建多个订阅器
    map_subs_.resize(num_maps_);
    for (int i = 0; i < num_maps_; i++) {
      std::string topic = "/neo_map_" + std::to_string(i);
      map_subs_[i] = nh.subscribe<msg::NeoMap>(
          topic, 1, 
          boost::bind(&MultiMapNode::mapCallback, this, _1, i));
      ROS_INFO("Subscribed to topic: %s", topic.c_str());
    }
  }

 private:
  void mapCallback(const msg::NeoMap::ConstPtr& msg, int map_id) {
    ROS_INFO("Received map from topic %d", map_id);
    try {
      // 使用NeoMapConverter将ROS消息转换为NeoMap对象
      auto neo_map = NeoMapConverter::fromROSMsg(*msg);
      
      // 更新全局地图
      gb_map_->Get(neo_map);
      
      ROS_INFO("Successfully updated global map with map %d", map_id);
    } catch (const std::exception& e) {
      ROS_ERROR("Error processing map %d: %s", map_id, e.what());
    }
  }

 private:
  int num_maps_;  // 订阅的地图数量
  std::shared_ptr<GBMap> gb_map_;  // 全局地图
  std::vector<ros::Subscriber> map_subs_;  // 地图话题订阅器
};

}  // namespace rgor

int main(int argc, char** argv) {
  ros::init(argc, argv, "multi_map_node");
  ros::NodeHandle nh;
  
  rgor::MultiMapNode node(nh);
  
  ROS_INFO("Multi map node started");
  ros::spin();
  
  return 0;
}