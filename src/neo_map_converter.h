/**
 * Copyright (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * RGOR (Relocalization with Generalized Object Recognition) is licensed
 * under Mulan PSL v2.
 */

#include <ros/ros.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/nil_generator.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_set>
#include <unordered_map>

#include "../rgor/include/common/NeoMap.h"
#include "../msg/NeoMap.h"
#include "../msg/NeoMapPoint.h"
#include "../msg/NeoKeyFrame.h"
#include "../msg/NeoKFMeasurement.h"

namespace rgor {

class NeoMapConverter {
public:
    /**
     * 将NeoMap对象转换为ROS消息
     * @param neo_map 待转换的NeoMap对象
     * @return 转换后的ROS消息
     */
    static msg::NeoMap toROSMsg(const std::shared_ptr<NeoMap>& neo_map) {
        msg::NeoMap msg;
        
        // 设置UUID
        msg.gb_uuid = boost::uuids::to_string(neo_map->GetUUID());
        
        // 转换地图点
        const auto& map_points = neo_map->GetMapPoints();
        for (const auto& mp_pair : map_points) {
            const auto& mp = mp_pair.second;
            msg::NeoMapPoint mp_msg;
            
            mp_msg.uuid = boost::uuids::to_string(mp->uuid);
            mp_msg.desc = mp->descriptor;
            mp_msg.pose.x = mp->position[0];
            mp_msg.pose.y = mp->position[1];
            mp_msg.pose.z = mp->position[2];
            mp_msg.scale = mp->scale;
            
            // 转换观测关键帧
            for (const auto& kf_uuid : mp->observations) {
                mp_msg.observations.push_back(boost::uuids::to_string(kf_uuid));
            }
            
            // 转换时间戳
            mp_msg.updated_at = std::chrono::system_clock::to_time_t(mp->updated_at);
            mp_msg.created_at = std::chrono::system_clock::to_time_t(mp->created_at);
            
            msg.mps.push_back(mp_msg);
        }
        
        // 转换关键帧
        const auto& keyframes = neo_map->GetKeyFrames();
        for (const auto& kf_pair : keyframes) {
            const auto& kf = kf_pair.second;
            msg::NeoKeyFrame kf_msg;
            
            kf_msg.uuid = boost::uuids::to_string(kf->uuid);
            
            // 相对位姿
            kf_msg.rotation_rel.x = kf->rotation_rel[0];
            kf_msg.rotation_rel.y = kf->rotation_rel[1];
            kf_msg.rotation_rel.z = kf->rotation_rel[2];
            kf_msg.rotation_rel.w = kf->rotation_rel[3];
            
            kf_msg.pose_rel.x = kf->position_rel[0];
            kf_msg.pose_rel.y = kf->position_rel[1];
            kf_msg.pose_rel.z = kf->position_rel[2];
            
            // 绝对位姿
            kf_msg.rotation_abs.x = kf->rotation_abs[0];
            kf_msg.rotation_abs.y = kf->rotation_abs[1];
            kf_msg.rotation_abs.z = kf->rotation_abs[2];
            kf_msg.rotation_abs.w = kf->rotation_abs[3];
            
            kf_msg.pose_abs.x = kf->position_abs[0];
            kf_msg.pose_abs.y = kf->position_abs[1];
            kf_msg.pose_abs.z = kf->position_abs[2];
            
            // 转换观测数据
            for (const auto& measure_pair : kf->measurements) {
                msg::NeoKFMeasurement m_msg;
                m_msg.mp_uuid = boost::uuids::to_string(measure_pair.first);
                // 可以在这里添加其他测量数据的转换
                kf_msg.measurement.push_back(m_msg);
            }
            
            // 前后关键帧
            kf_msg.pre_kf = boost::uuids::to_string(kf->pre_keyframe);
            if (kf->next_keyframe != boost::uuids::nil_generator()()) {
                kf_msg.next_kf = boost::uuids::to_string(kf->next_keyframe);
            }
            
            // 时间戳
            kf_msg.updated_at = std::chrono::system_clock::to_time_t(kf->updated_at);
            kf_msg.created_at = std::chrono::system_clock::to_time_t(kf->created_at);
            
            msg.kfs.push_back(kf_msg);
        }
        
        return msg;
    }
    
    /**
     * 将ROS消息转换为NeoMap对象
     * @param msg 待转换的ROS消息
     * @return 转换后的NeoMap对象
     */
    static std::shared_ptr<NeoMap> fromROSMsg(const msg::NeoMap& msg) {
        auto neo_map = std::make_shared<NeoMap>(boost::uuids::string_generator()(msg.gb_uuid));
        
        // 转换地图点
        for (const auto& mp : msg.mps) {
            neo_map->AddMapPoint(
                boost::uuids::string_generator()(mp.uuid),
                mp.desc,  // 描述子
                Eigen::Vector3f(mp.pose.x, mp.pose.y, mp.pose.z),  // 位置
                mp.scale,  // 尺度
                std::unordered_set<boost::uuids::uuid>(  // 观测到该点的关键帧集合
                    mp.observations.begin(), 
                    mp.observations.end()
                ),
                std::chrono::system_clock::from_time_t(mp.updated_at),  // 更新时间
                std::chrono::system_clock::from_time_t(mp.created_at)  // 创建时间
            );
        }
        
        // 转换关键帧
        for (const auto& kf : msg.kfs) {
            // 构建关键帧的观测数据
            std::unordered_map<boost::uuids::uuid, std::unordered_map<std::string, float>> measurements;
            for (const auto& m : kf.measurement) {
                measurements[boost::uuids::string_generator()(m.mp_uuid)] = 
                    std::unordered_map<std::string, float>();
            }

            neo_map->AddKeyFrame(
                boost::uuids::string_generator()(kf.uuid),  // 关键帧ID
                Eigen::Vector4f(  // 相对旋转
                    kf.rotation_rel.x,
                    kf.rotation_rel.y,
                    kf.rotation_rel.z,
                    kf.rotation_rel.w
                ),
                Eigen::Vector3f(  // 相对平移
                    kf.pose_rel.x,
                    kf.pose_rel.y,
                    kf.pose_rel.z
                ),
                Eigen::Vector4f(  // 绝对旋转
                    kf.rotation_abs.x,
                    kf.rotation_abs.y,
                    kf.rotation_abs.z,
                    kf.rotation_abs.w
                ),
                Eigen::Vector3f(  // 绝对平移
                    kf.pose_abs.x,
                    kf.pose_abs.y,
                    kf.pose_abs.z
                ),
                measurements,  // 观测数据
                boost::uuids::string_generator()(kf.pre_kf),  // 前一关键帧
                kf.next_kf.empty() ? boost::uuids::nil_generator()() 
                                : boost::uuids::string_generator()(kf.next_kf),  // 下一关键帧
                std::chrono::system_clock::from_time_t(kf.updated_at),  // 更新时间
                std::chrono::system_clock::from_time_t(kf.created_at),  // 创建时间
                true  // 是否为关键帧
            );
        }
        
        return neo_map;
    }
};

}  // namespace rgor