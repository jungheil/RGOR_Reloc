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

#include <ros/ros.h>
#include "common/NeoMap.h"
#include "msg/NeoMap.h"
#include "msg/NeoMapPoint.h"
#include "msg/NeoKeyFrame.h"
#include "msg/NeoKFMeasurement.h"
#include "msg/NeoPose.h"
#include "msg/NeoRotation.h"
#include "msg/NeoScale.h"

class NeoMapConverterNode {
public:
    NeoMapConverterNode(ros::NodeHandle &nh) {
        map_pub_ = nh.advertise<msg::NeoMap>("/rgor/neo_map", 1);
    }

    void publishMap(const std::shared_ptr<rgor::NeoMap> &neo_map) {
        msg::NeoMap ros_map;
        convertNeoMapToMsg(neo_map, ros_map);
        map_pub_.publish(ros_map);
    }

private:
    ros::Publisher map_pub_;

    // 转换NeoMapPoint到ROS消息
    void convertMapPoint(const rgor::NeoMapPoint::Ptr &mp, msg::NeoMapPoint &ros_mp) {
        // UUID
        ros_mp.uuid = uuids::to_string(mp->get_uuid());

        // 描述子
        auto desc = mp->get_descriptor();
        std::string desc_str;
        desc_str.resize(desc.size() * sizeof(float));
        std::memcpy(&desc_str[0], desc.data(), desc.size() * sizeof(float));
        ros_mp.desc = desc_str;

        // 位置
        auto pos = mp->get_pos();
        ros_mp.pose.x = pos[0];
        ros_mp.pose.y = pos[1];
        ros_mp.pose.z = pos[2];

        // 尺度
        auto scale = mp->get_scale();
        ros_mp.scale.s = scale.first;
        ros_mp.scale.l = scale.second;

        // 观测帧
        for (const auto &obs : mp->get_observations()) {
            ros_mp.observations.push_back(uuids::to_string(obs));
        }

        // 时间戳
        ros_mp.updated_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            mp->get_updated_at().time_since_epoch()).count();
        ros_mp.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            mp->get_created_at().time_since_epoch()).count();
    }

    // 转换NeoKeyFrame到ROS消息
    void convertKeyFrame(const rgor::NeoKeyFrame::Ptr &kf, msg::NeoKeyFrame &ros_kf) {
        // UUID
        ros_kf.uuid = uuids::to_string(kf->get_uuid());

        // 相对位姿
        auto rel_t = kf->get_rel_t_cw();
        ros_kf.pose_rel.x = rel_t[0];
        ros_kf.pose_rel.y = rel_t[1];
        ros_kf.pose_rel.z = rel_t[2];

        auto rel_r = kf->get_rel_r_cw();
        ros_kf.rotation_rel.w = rel_r[0];
        ros_kf.rotation_rel.x = rel_r[1];
        ros_kf.rotation_rel.y = rel_r[2];
        ros_kf.rotation_rel.z = rel_r[3];

        // 绝对位姿
        auto abs_t = kf->get_abs_t_cw();
        ros_kf.pose_abs.x = abs_t[0];
        ros_kf.pose_abs.y = abs_t[1];
        ros_kf.pose_abs.z = abs_t[2];

        auto abs_r = kf->get_abs_r_cw();
        ros_kf.rotation_abs.w = abs_r[0];
        ros_kf.rotation_abs.x = abs_r[1];
        ros_kf.rotation_abs.y = abs_r[2];
        ros_kf.rotation_abs.z = abs_r[3];

        // 相邻帧
        ros_kf.pre_kf = uuids::to_string(kf->get_pre_kf());
        ros_kf.next_kf = uuids::to_string(kf->get_next_kf());

        // 观测到的MapPoints
        for (const auto &[mp_uuid, mp_pos] : kf->get_measurement_mps()) {
            msg::NeoKFMeasurement measurement;
            measurement.uuid = uuids::to_string(mp_uuid);
            measurement.pose.x = mp_pos[0];
            measurement.pose.y = mp_pos[1];
            measurement.pose.z = mp_pos[2];
            ros_kf.measurement.push_back(measurement);
        }

        // 时间戳
        ros_kf.updated_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            kf->get_updated_at().time_since_epoch()).count();
        ros_kf.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            kf->get_created_at().time_since_epoch()).count();
    }

    // 转换整个NeoMap到ROS消息
    void convertNeoMapToMsg(const std::shared_ptr<rgor::NeoMap> &neo_map, msg::NeoMap &ros_map) {
        // 设置头信息
        ros_map.header.stamp = ros::Time::now();
        ros_map.header.frame_id = "map";

        // 转换所有MapPoints
        auto mp_uuids = neo_map->GetMPSUUID();
        for (const auto &uuid : mp_uuids) {
            msg::NeoMapPoint ros_mp;
            auto mp = neo_map->GetMPByUUID(uuid);
            convertMapPoint(mp, ros_mp);
            ros_map.mps.push_back(ros_mp);
        }

        // 转换所有KeyFrames
        auto kf_uuids = neo_map->GetKFSUUID();
        for (const auto &uuid : kf_uuids) {
            msg::NeoKeyFrame ros_kf;
            auto kf = neo_map->GetKFByUUID(uuid);
            convertKeyFrame(kf, ros_kf);
            ros_map.kfs.push_back(ros_kf);
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "neo_map_converter_node");
    ros::NodeHandle nh("~");

    NeoMapConverterNode converter(nh);
    ros::spin();

    return 0;
}