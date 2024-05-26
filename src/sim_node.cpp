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

/**
 * 　　　┏┓　　　┏┓
 * 　　┏┛┻━━━┛┻┓
 * 　　┃　　　　　　　 ┃
 * 　　┃　　　━　　　 ┃
 * 　　┃　┳┛　┗┳　┃
 * 　　┃　　　　　　　 ┃
 * 　　┃　　　┻　　　 ┃
 * 　　┃　　　　　　　 ┃
 * 　　┗━┓　　　┏━┛Codes are far away from bugs with the animal protecting
 * 　　　　┃　　　┃ 神兽保佑, 代码无bug
 * 　　　　┃　　　┃
 * 　　　　┃　　　┗━━━┓
 * 　　　　┃　　　　　 ┣┓
 * 　　　　┃　　　　 ┏┛
 * 　　　　┗┓┓┏━┳┓┏┛
 * 　　　　　┃┫┫　┃┫┫
 * 　　　　　┗┻┛　┗┻┛
 */

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/cache.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Path.h>
#include <rgor_sys/DetectionObj.h>
#include <rgor_sys/MultiDetectionObj.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Header.h>
#include <std_srvs/Empty.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "CameraPoseVisualization.h"
#include "RGOR.h"
#include "utils/draw_sphere.h"
#include "visualization.h"

struct SysCB {
  ros::Publisher arimg_pub;
  ros::Publisher cpimg_pub_;
  ros::Publisher map_pub_;
  ros::Publisher obj_pub_;
  ros::Publisher path_pub_;
  ros::Publisher cam_pose_pub_;
  ros::Publisher rloc_pose_pub_;
  ros::ServiceServer save_map_srv_;

  // 记录实验数据
  size_t kf_count = 0;
  size_t save_img_idx = 0;

  GetMarkerArray get_marker;
  std::shared_ptr<message_filters::Cache<sensor_msgs::Image>> rgb_cache_;
  std::shared_ptr<message_filters::Cache<sensor_msgs::Image>> depth_cache_;
  std::shared_ptr<message_filters::Cache<geometry_msgs::PoseStamped>>
      pose_cache_;
  std::shared_ptr<rgor::System> sys_;
  std::shared_ptr<rgor::Camera> cam_;

  nav_msgs::Path path_;
  CameraPoseVisualization rloc_pose_visual_{0, 0, 1, 1};
  CameraPoseVisualization cam_pose_visual_{0, 1, 0, 1};

  std::filesystem::path output_path_;
  std::string task_name_;
  bool save_data_;

  typedef void result_type;
  SysCB(ros::NodeHandle &nh,
        std::shared_ptr<message_filters::Cache<sensor_msgs::Image>> rgb_cache,
        std::shared_ptr<message_filters::Cache<sensor_msgs::Image>> depth_cache,
        std::shared_ptr<message_filters::Cache<geometry_msgs::PoseStamped>>
            pose_cache,
        std::shared_ptr<rgor::System> sys, std::shared_ptr<rgor::Camera> cam,
        std::filesystem::path output_path, bool save_data = false) {
    arimg_pub = nh.advertise<sensor_msgs::Image>("/rgor/ar_img", 1);
    cpimg_pub_ = nh.advertise<sensor_msgs::Image>("/rgor/kp_img", 1);
    map_pub_ = nh.advertise<sensor_msgs::PointCloud>("/rgor/map", 1);
    obj_pub_ =
        nh.advertise<visualization_msgs::MarkerArray>("/rgor/object_marker", 1);
    path_pub_ = nh.advertise<nav_msgs::Path>("/rgor/path", 1);
    cam_pose_pub_ =
        nh.advertise<visualization_msgs::MarkerArray>("/rgor/cam_pose", 1);
    rloc_pose_pub_ =
        nh.advertise<visualization_msgs::MarkerArray>("/rgor/rloc_pose", 1);
    save_map_srv_ =
        nh.advertiseService("/rgor/save_map", &SysCB::SaveMapCB, this);

    get_marker = GetMarkerArray();
    rgb_cache_ = std::move(rgb_cache);
    depth_cache_ = std::move(depth_cache);
    pose_cache_ = std::move(pose_cache);
    sys_ = std::move(sys);
    cam_ = std::move(cam);

    cam_pose_visual_.setScale(0.35);
    cam_pose_visual_.setLineWidth(0.08);
    rloc_pose_visual_.setScale(0.35);
    rloc_pose_visual_.setLineWidth(0.08);

    output_path_ = std::move(output_path);
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y_%m_%d_%H_%M_%S");
    task_name_ = ss.str();
    save_data_ = save_data;
  };

  bool SaveMapCB(std_srvs::Empty::Request &req,
                 std_srvs::Empty::Response &res) {
    std::string map_name = output_path_ / "rgor_map" / (task_name_ + ".bin");
    std::filesystem::create_directories(output_path_ / "rgor_map");
    sys_->SaveMap(map_name);
    return true;
  }

  void operator()(const rgor_sys::MultiDetectionObjConstPtr &object_data) {
    ROS_INFO("sys_cb start");
    auto time = object_data->header.stamp + ros::Duration(0.001);

    auto rgb_data = rgb_cache_->getElemBeforeTime(time);
    auto depth_data = depth_cache_->getElemBeforeTime(time);
    auto pose_data = pose_cache_->getElemBeforeTime(time);
    if (pose_data == nullptr ||
        object_data->header.stamp - pose_data->header.stamp >
            ros::Duration(0.05)) {
      ROS_WARN("No pose data found.");
      return;
    }

    if (rgb_data == nullptr || depth_data == nullptr) {
      ROS_WARN("No rgb or depth data found.");
      return;
    } else if (rgb_data->header.stamp != object_data->header.stamp &&
               depth_data->header.stamp != object_data->header.stamp) {
      ROS_WARN("Time stamp not match. rgb: %f, depth: %f, obj: %f",
               rgb_data->header.stamp.toSec(), depth_data->header.stamp.toSec(),
               object_data->header.stamp.toSec());
      return;
    }

    cv_bridge::CvImagePtr rgb_ptr;
    cv_bridge::CvImagePtr depth_ptr;

    try {
      rgb_ptr =
          cv_bridge::toCvCopy(rgb_data, sensor_msgs::image_encodings::BGR8);
      depth_ptr = cv_bridge::toCvCopy(depth_data,
                                      sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat src = rgb_ptr->image.clone();

    for (const auto &p : object_data->obj) {
      int x = p.x;
      int y = p.y;
      cv::circle(src, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    }

    cv_bridge::CvImage cv_image;
    cv_image.image = src;
    cv_image.encoding = "bgr8";
    cpimg_pub_.publish(cv_image.toImageMsg());

    time_t timestamp = object_data->header.stamp.toSec();
    Eigen::Vector4f r_cw{static_cast<float>(pose_data->pose.orientation.x),
                         static_cast<float>(pose_data->pose.orientation.y),
                         static_cast<float>(pose_data->pose.orientation.z),
                         static_cast<float>(pose_data->pose.orientation.w)};

    Eigen::Vector3f t_cw{static_cast<float>(pose_data->pose.position.x),
                         static_cast<float>(pose_data->pose.position.y),
                         static_cast<float>(pose_data->pose.position.z)};

    std::vector<rgor::KeyPoint::Ptr> kps;
    for (size_t i = 0; i < object_data->obj.size(); ++i) {
      float p_x = object_data->obj[i].x;
      float p_y = object_data->obj[i].y;

      if (p_x < 0 || p_x >= src.cols || p_y < 0 || p_y >= src.rows) {
        ROS_WARN("Invalid keypoint position.");
      }

      float p_depth = depth_ptr->image.at<uint16_t>(int(p_y), int(p_x));
      float x_scaled = object_data->obj[i].w;
      float y_scaled = object_data->obj[i].h;

      std::vector<float> desc_data(object_data->obj[i].class_prob.begin(),
                                   object_data->obj[i].class_prob.end());
      auto descriptor =
          Eigen::Map<Eigen::VectorXf>(desc_data.data(), desc_data.size());
      kps.emplace_back(std::make_shared<rgor::KeyPoint>(
          p_x, p_y, p_depth, x_scaled, y_scaled, descriptor));
    }
    Eigen::Vector4f RR;
    Eigen::Vector3f RT;
    auto add_kf = sys_->AddFrame({cam_, r_cw, t_cw, timestamp, kps}, RR, RT);

    // 纪录实验数据
    if (save_data_ && add_kf) {
      std::filesystem::path img_path = output_path_ / "rgor_img" / task_name_;
      std::filesystem::create_directories(img_path);

      std::string image_name = std::to_string(save_img_idx++);
      while (image_name.size() < 5) {
        image_name = "0" + image_name;
      }

      cv::imwrite(img_path / (image_name + ".jpg"), rgb_ptr->image);
    }

    if (RR[3] > 0) {
      Eigen::Vector3d P = {RT[0], RT[1], RT[2]};
      if (P.norm() < 1.5) {
        Eigen::Vector3d P_s = {t_cw[0], t_cw[1], t_cw[2]};
        P = P + P_s;
        Eigen::Quaterniond R_s = {
            pose_data->pose.orientation.x, pose_data->pose.orientation.y,
            pose_data->pose.orientation.z, pose_data->pose.orientation.w};
        Eigen::Quaterniond R = {RR[0], RR[1], RR[2], RR[3]};
        R_s = R * R_s;
        auto RSM = R_s.toRotationMatrix();
        Eigen::Matrix3d RCW;
        RCW << -1, 0, 0, 0, 0, 1, 0, 1, 0;
        RSM = RCW * RSM;
        R_s = Eigen::Quaterniond(RSM);
        R = R_s;

        rloc_pose_visual_.add_pose(P, R);
        std_msgs::Header cam_pose_header;
        cam_pose_header.stamp = object_data->header.stamp;
        cam_pose_header.frame_id = "map";
        rloc_pose_visual_.publish_by(rloc_pose_pub_, cam_pose_header);

        // 记录实验数据
        if (save_data_) {
          std::ofstream outfile(
              output_path_ / "rgor_data" / (task_name_ + ".txt"),
              std::ios::app);
          outfile << kf_count << " / " << RR.transpose() << " / "
                  << RT.transpose() << std::endl;
          outfile.close();
        }
      }
    } else if (RR[3] < 0) {
      kf_count++;
    }

    Eigen::Vector3d P = {pose_data->pose.position.x, pose_data->pose.position.y,
                         pose_data->pose.position.z};
    Eigen::Quaterniond R = {
        pose_data->pose.orientation.x, pose_data->pose.orientation.y,
        pose_data->pose.orientation.z, pose_data->pose.orientation.w};
    auto RSM = R.toRotationMatrix();
    Eigen::Matrix3d R90;
    R90 << 1, 0, 0, 0, 0, 1, 0, -1, 0;
    RSM = R90 * RSM;
    R = Eigen::Quaterniond(RSM);
    cam_pose_visual_.reset();
    cam_pose_visual_.add_pose(P, R);
    std_msgs::Header cam_pose_header;
    cam_pose_header.stamp = object_data->header.stamp;
    cam_pose_header.frame_id = "map";
    cam_pose_visual_.publish_by(cam_pose_pub_, cam_pose_header);

    geometry_msgs::PoseStamped pose;
    pose.pose = pose_data->pose;
    pose.header.stamp = object_data->header.stamp;
    pose.header.frame_id = "map";
    path_.header.stamp = object_data->header.stamp;
    path_.header.frame_id = "map";
    path_.poses.push_back(pose);
    path_pub_.publish(path_);

    // publish map pointcloud
    auto mps = sys_->get_map()->get_mps();

    sensor_msgs::PointCloud mp_cloud;
    mp_cloud.header.frame_id = "map";
    mp_cloud.header.stamp = object_data->header.stamp;
    mp_cloud.points.resize(mps.size());
    mp_cloud.channels.resize(1);
    mp_cloud.channels[0].name = "intensity";
    mp_cloud.channels[0].values.resize(mps.size());

    for (size_t i = 0; i < mps.size(); ++i) {
      mp_cloud.points[i].x = mps[i]->get_pos()[0];
      mp_cloud.points[i].y = mps[i]->get_pos()[1];
      mp_cloud.points[i].z = mps[i]->get_pos()[2];
      mp_cloud.channels[0].values[i] = 1;
    }

    map_pub_.publish(mp_cloud);

    // publish map marker
    auto marker_array = get_marker(mps, object_data->header.stamp);
    obj_pub_.publish(marker_array);

    // publish ar image
    auto ar_src = rgb_ptr->image.clone();
    auto ar_mp = sys_->GetMPInViews(r_cw, t_cw, cam_);
    for (const auto &mp : ar_mp) {
      float scale = mp->get_scale().first;
      auto [r, g, b] = GetColor(mp->get_descriptor());

      Eigen::Quaternionf q(r_cw);
      Eigen::Matrix3f R_wc = q.toRotationMatrix().transpose();
      auto pos_c = R_wc * (mp->get_pos() - t_cw);
      cv::Mat K_cv = cv::Mat::eye(3, 3, CV_32F);
      K_cv.at<float>(0, 0) = cam_->K(0, 0);
      K_cv.at<float>(1, 1) = cam_->K(1, 1);
      K_cv.at<float>(0, 2) = cam_->K(0, 2);
      K_cv.at<float>(1, 2) = cam_->K(1, 2);

      drawSphereMesh(ar_src, cv::Point3f(pos_c[0], pos_c[1], pos_c[2]),
                     scale / 2, K_cv, cv::Scalar(b * 255, g * 255, r * 255), 1,
                     4);
    }
    cv_bridge::CvImage ar_cv_image;
    ar_cv_image.image = ar_src;
    ar_cv_image.encoding = "bgr8";
    arimg_pub.publish(ar_cv_image.toImageMsg());
  }
};

void GetParams(const ros::NodeHandle &nh, rgor::TrackingParams &tracking_params,
               rgor::MatcherParams &g_matcher_params,
               rgor::MatcherParams &l_matcher_params,
               rgor::MappingParams &mapping_params,
               rgor::RelocationParams &relocation_params) {
  tracking_params.lmps_cache_size =
      nh.param<int>("tracking/lmps_cache_size", 4);
  tracking_params.tmps_cache_size =
      nh.param<int>("tracking/tmps_cache_size", 16);
  tracking_params.obs_threshold = nh.param<int>("tracking/obs_threshold", 8);
  tracking_params.obs_ratio_threshold =
      nh.param<float>("tracking/obs_ratio_threshold", 0.6);
  tracking_params.nkf_t_threshold =
      nh.param<float>("tracking/nkf_t_threshold", 0.5);
  tracking_params.nkf_r_threshold =
      nh.param<float>("tracking/nkf_r_threshold", 0.5);
  tracking_params.small_object_pixel =
      nh.param<int>("tracking/small_object_pixel", 5);
  tracking_params.small_object_scale =
      nh.param<float>("tracking/small_object_scale", 0.1);

  g_matcher_params.knn_search_num =
      nh.param<int>("g_matcher/knn_search_num", 8);
  g_matcher_params.neighbour_radius =
      nh.param<float>("g_matcher/neighbour_radius", 0.2);
  g_matcher_params.desc_dist_threshold =
      nh.param<float>("g_matcher/desc_dist_threshold", 0.5);
  g_matcher_params.scale_dist_threshold =
      nh.param<float>("g_matcher/scale_dist_threshold", 0.6);
  g_matcher_params.dist_scale_ratio =
      nh.param<float>("g_matcher/dist_scale_ratio", 0.5);

  l_matcher_params.knn_search_num =
      nh.param<int>("l_matcher/knn_search_num", 8);
  l_matcher_params.neighbour_radius =
      nh.param<float>("l_matcher/neighbour_radius", 0.1);
  l_matcher_params.desc_dist_threshold =
      nh.param<float>("l_matcher/desc_dist_threshold", 0.6);
  l_matcher_params.scale_dist_threshold =
      nh.param<float>("l_matcher/scale_dist_threshold", 0.7);
  l_matcher_params.dist_scale_ratio =
      nh.param<float>("l_matcher/dist_scale_ratio", 0.5);

  mapping_params.dist_threshold =
      nh.param<float>("mapping/dist_threshold", 0.2);
  mapping_params.dist_scale_ratio =
      nh.param<float>("mapping/dist_scale_ratio", 0.7);
  mapping_params.desc_score_threshold_1 =
      nh.param<float>("mapping/desc_score_threshold_1", 0.5);
  mapping_params.scale_score_threshold_1 =
      nh.param<float>("mapping/scale_score_threshold_1", 0.8);
  mapping_params.desc_score_threshold_2 =
      nh.param<float>("mapping/desc_score_threshold_2", 0.5);
  mapping_params.scale_score_threshold_2 =
      nh.param<float>("mapping/scale_score_threshold_2", 0.8);

  relocation_params.neighbour_radius =
      nh.param<float>("relocation/neighbour_radius", 3);
  relocation_params.scale_score_threshold =
      nh.param<float>("relocation/scale_score_threshold", 0.8);
  relocation_params.desc_score_threshold =
      nh.param<float>("relocation/desc_score_threshold", 0.5);
  relocation_params.pair_score_threshold =
      nh.param<float>("relocation/pair_score_threshold", 0.5);
  relocation_params.best_match_threshold =
      nh.param<float>("relocation/best_match_threshold", 2);
  relocation_params.fine_dist_threshold =
      nh.param<float>("relocation/fine_dist_threshold", 0.3);
  relocation_params.fine_desc_threshold =
      nh.param<float>("relocation/fine_desc_threshold", 0.3);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "sim_node");
  ros::NodeHandle nh("~");

  rgor::TrackingParams tracking_params;
  rgor::MatcherParams g_matcher_params;
  rgor::MatcherParams l_matcher_params;

  rgor::MappingParams mapping_params;
  rgor::RelocationParams relocation_params;

  GetParams(nh, tracking_params, g_matcher_params, l_matcher_params,
            mapping_params, relocation_params);

  float fx = nh.param<float>("camera/fx", 600);
  float fy = nh.param<float>("camera/fy", 600);
  float cx = nh.param<float>("camera/cx", 320);
  float cy = nh.param<float>("camera/cy", 240);
  int width = nh.param<int>("camera/width", 640);
  int height = nh.param<int>("camera/height", 480);
  float min_depth = nh.param<float>("camera/min_depth", 0.5);
  float max_depth = nh.param<float>("camera/max_depth", 8.0);
  float valid_wh_ratio = nh.param<float>("camera/valid_wh_ratio", 0.95);

  std::string ros_rgb_topic =
      nh.param<std::string>("ros/rgb_topic", "/rgor/camera/color");
  std::string ros_depth_topic =
      nh.param<std::string>("ros/depth_topic", "rgor/camera/depth");
  std::string ros_pose_topic =
      nh.param<std::string>("ros/pose_topic", "/rgor/camera/color");
  std::string ros_object_topic =
      nh.param<std::string>("ros/object_topic", "rgor/camera/object");

  auto output_path = nh.param<std::string>("ros/output_path", "~/rgor_data/");

  std::filesystem::create_directories(output_path);

  std::cout << "ros_rgb_topic: " << ros_rgb_topic << std::endl;
  std::cout << "ros_depth_topic: " << ros_depth_topic << std::endl;
  std::cout << "ros_pose_topic: " << ros_pose_topic << std::endl;
  std::cout << "ros_object_topic: " << ros_object_topic << std::endl;

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, ros_rgb_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, ros_depth_topic,
                                                            1);
  message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub(
      nh, ros_pose_topic, 1);
  message_filters::Subscriber<rgor_sys::MultiDetectionObj> obj_sub(
      nh, ros_object_topic, 1);

  Eigen::Matrix3f K;
  Eigen::VectorXf dist_coeffs(5);
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  dist_coeffs << 0, 0, 0, 0, 0;
  auto cam = std::make_shared<rgor::Camera>(
      K, dist_coeffs, width, height, min_depth, max_depth, valid_wh_ratio);

  auto sys = std::make_shared<rgor::System>(tracking_params, g_matcher_params,
                                            l_matcher_params, mapping_params,
                                            relocation_params);

  auto rgb_cache = std::make_shared<message_filters::Cache<sensor_msgs::Image>>(
      rgb_sub, 4096);
  auto depth_cache =
      std::make_shared<message_filters::Cache<sensor_msgs::Image>>(depth_sub,
                                                                   4096);
  auto pose_cache =
      std::make_shared<message_filters::Cache<geometry_msgs::PoseStamped>>(
          pose_sub, 8192);
  SysCB sys_cb =
      SysCB(nh, rgb_cache, depth_cache, pose_cache, sys, cam, output_path);

  obj_sub.registerCallback(std::bind(sys_cb, std::placeholders::_1));

  ROS_INFO("rgor_sys_node has started.");

  ros::spin();

  return 0;
}
