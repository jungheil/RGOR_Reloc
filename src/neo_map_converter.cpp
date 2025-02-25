#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "proto/PMap.pb.h"
#include "common/NeoMap.h"

// ROS消息头文件
#include "msg/NeoMap.h"
#include "msg/NeoMapPoint.h"
#include "msg/NeoKeyFrame.h"
#include "msg/NeoKFMeasurement.h"
#include "msg/NeoPose.h"
#include "msg/NeoRotation.h"
#include "msg/NeoScale.h"

namespace rgor {

// Proto转ROS消息的辅助函数
void convertNeoPose(const NeoPose& proto_pose, msg::NeoPose& ros_pose) {
    ros_pose.x = proto_pose.x();
    ros_pose.y = proto_pose.y();
    ros_pose.z = proto_pose.z();
}

void convertNeoRotation(const NeoRotation& proto_rot, msg::NeoRotation& ros_rot) {
    ros_rot.w = proto_rot.w();
    ros_rot.x = proto_rot.x();
    ros_rot.y = proto_rot.y();
    ros_rot.z = proto_rot.z();
}

void convertNeoScale(const NeoScale& proto_scale, msg::NeoScale& ros_scale) {
    ros_scale.s = proto_scale.s();
    ros_scale.l = proto_scale.l();
}

void convertNeoKFMeasurement(const NeoKFMeasurement& proto_meas, msg::NeoKFMeasurement& ros_meas) {
    ros_meas.uuid = std::string(proto_meas.uuid().begin(), proto_meas.uuid().end());
    convertNeoPose(proto_meas.pose(), ros_meas.pose);
}

void convertNeoMapPoint(const NeoMapPoint& proto_mp, msg::NeoMapPoint& ros_mp) {
    // 转换基本字段
    ros_mp.uuid = std::string(proto_mp.uuid().begin(), proto_mp.uuid().end());
    ros_mp.desc = std::string(proto_mp.desc().begin(), proto_mp.desc().end());
    
    // 转换scale
    convertNeoScale(proto_mp.scale(), ros_mp.scale);
    
    // 转换pose
    convertNeoPose(proto_mp.pose(), ros_mp.pose);
    
    // 转换observations数组
    ros_mp.observations.clear();
    for (const auto& obs : proto_mp.observations()) {
        ros_mp.observations.push_back(std::string(obs.begin(), obs.end()));
    }
    
    // 转换时间戳
    ros_mp.updated_at = proto_mp.updated_at();
    ros_mp.created_at = proto_mp.created_at();
}

void convertNeoKeyFrame(const NeoKeyFrame& proto_kf, msg::NeoKeyFrame& ros_kf) {
    // 转换基本字段
    ros_kf.uuid = std::string(proto_kf.uuid().begin(), proto_kf.uuid().end());
    
    // 转换位姿
    convertNeoPose(proto_kf.pose_rel(), ros_kf.pose_rel);
    convertNeoRotation(proto_kf.rotation_rel(), ros_kf.rotation_rel);
    convertNeoPose(proto_kf.pose_abs(), ros_kf.pose_abs);
    convertNeoRotation(proto_kf.rotation_abs(), ros_kf.rotation_abs);
    
    // 转换关联帧
    ros_kf.pre_kf = std::string(proto_kf.pre_kf().begin(), proto_kf.pre_kf().end());
    ros_kf.next_kf = std::string(proto_kf.next_kf().begin(), proto_kf.next_kf().end());
    
    // 转换观测点
    ros_kf.measurement.clear();
    for (const auto& meas : proto_kf.measurement()) {
        msg::NeoKFMeasurement ros_meas;
        convertNeoKFMeasurement(meas, ros_meas);
        ros_kf.measurement.push_back(ros_meas);
    }
    
    // 转换时间戳
    ros_kf.updated_at = proto_kf.updated_at();
    ros_kf.created_at = proto_kf.created_at();
}

void convertNeoMap(const NeoMap& proto_map, msg::NeoMap& ros_map) {
    // 设置头信息
    ros_map.header.stamp = ros::Time::now();
    ros_map.header.frame_id = "map";
    
    // 转换地图点
    ros_map.mps.clear();
    for (const auto& mp : proto_map.mps()) {
        msg::NeoMapPoint ros_mp;
        convertNeoMapPoint(mp, ros_mp);
        ros_map.mps.push_back(ros_mp);
    }
    
    // 转换关键帧
    ros_map.kfs.clear();
    for (const auto& kf : proto_map.kfs()) {
        msg::NeoKeyFrame ros_kf;
        convertNeoKeyFrame(kf, ros_kf);
        ros_map.kfs.push_back(ros_kf);
    }
}

// ROS消息转Proto的函数
void convertToProtoNeoPose(const msg::NeoPose& ros_pose, NeoPose* proto_pose) {
    proto_pose->set_x(ros_pose.x);
    proto_pose->set_y(ros_pose.y);
    proto_pose->set_z(ros_pose.z);
}

void convertToProtoNeoRotation(const msg::NeoRotation& ros_rot, NeoRotation* proto_rot) {
    proto_rot->set_w(ros_rot.w);
    proto_rot->set_x(ros_rot.x);
    proto_rot->set_y(ros_rot.y);
    proto_rot->set_z(ros_rot.z);
}

void convertToProtoNeoScale(const msg::NeoScale& ros_scale, NeoScale* proto_scale) {
    proto_scale->set_s(ros_scale.s);
    proto_scale->set_l(ros_scale.l);
}

void convertToProtoNeoMap(const msg::NeoMap& ros_map, NeoMap* proto_map) {
    proto_map->Clear();
    
    // 转换地图点
    for (const auto& ros_mp : ros_map.mps) {
        auto* proto_mp = proto_map->add_mps();
        proto_mp->set_uuid(ros_mp.uuid);
        proto_mp->set_desc(ros_mp.desc);
        
        // 转换scale
        auto* proto_scale = proto_mp->mutable_scale();
        convertToProtoNeoScale(ros_mp.scale, proto_scale);
        
        // 转换pose
        auto* proto_pose = proto_mp->mutable_pose();
        convertToProtoNeoPose(ros_mp.pose, proto_pose);
        
        // 转换observations
        for (const auto& obs : ros_mp.observations) {
            proto_mp->add_observations(obs);
        }
        
        // 转换时间戳
        proto_mp->set_updated_at(ros_mp.updated_at);
        proto_mp->set_created_at(ros_mp.created_at);
    }
    
    // 转换关键帧
    for (const auto& ros_kf : ros_map.kfs) {
        auto* proto_kf = proto_map->add_kfs();
        proto_kf->set_uuid(ros_kf.uuid);
        
        // 转换位姿
        auto* proto_pose_rel = proto_kf->mutable_pose_rel();
        convertToProtoNeoPose(ros_kf.pose_rel, proto_pose_rel);
        
        auto* proto_rot_rel = proto_kf->mutable_rotation_rel();
        convertToProtoNeoRotation(ros_kf.rotation_rel, proto_rot_rel);
        
        auto* proto_pose_abs = proto_kf->mutable_pose_abs();
        convertToProtoNeoPose(ros_kf.pose_abs, proto_pose_abs);
        
        auto* proto_rot_abs = proto_kf->mutable_rotation_abs();
        convertToProtoNeoRotation(ros_kf.rotation_abs, proto_rot_abs);
        
        // 设置关联帧
        proto_kf->set_pre_kf(ros_kf.pre_kf);
        proto_kf->set_next_kf(ros_kf.next_kf);
        
        // 转换测量信息
        for (const auto& ros_meas : ros_kf.measurement) {
            auto* proto_meas = proto_kf->add_measurement();
            proto_meas->set_uuid(ros_meas.uuid);
            auto* proto_meas_pose = proto_meas->mutable_pose();
            convertToProtoNeoPose(ros_meas.pose, proto_meas_pose);
        }
        
        // 转换时间戳
        proto_kf->set_updated_at(ros_kf.updated_at);
        proto_kf->set_created_at(ros_kf.created_at);
    }
}

} // namespace rgor