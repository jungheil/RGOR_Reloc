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

#include "Optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <chrono>
#include <memory>
#include <set>
#include <unordered_map>

#include "g2o/core/eigen_types.h"
#include "g2o/core/robust_kernel_impl.h"
#include "include/common/Frame.h"

namespace rgor {

double DepthStdDev(double d)
// standard deviation of depth d  in meter
{
  double c1, c2, c3;
  c1 = 2.73e-1;
  c2 = 7.4e-2;
  c3 = -5.8e-2;

  return c1 * d * d + c2 * d + c3;
}

Eigen::Matrix3f GetPoint3dCov(Eigen::Vector3f pt, double fx, double fy,
                              double cu, double cv) {
  double sigma_impt = 1; // std dev of image sample point

  Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
  J(0, 0) = pt(2) / fx;
  J(0, 2) = pt(0) / pt(2);
  J(1, 1) = pt(2) / fy;
  J(1, 2) = pt(1) / pt(2);
  J(2, 2) = 1;

  Eigen::Matrix3f cov_g_d = Eigen::Matrix3f::Zero();
  cov_g_d(0, 0) = sigma_impt * sigma_impt;
  cov_g_d(1, 1) = sigma_impt * sigma_impt;
  cov_g_d(2, 2) = DepthStdDev(pt(2)) * DepthStdDev(pt(2));

  Eigen::Matrix3f cov = J * cov_g_d * J.transpose();

  return cov;
}

std::pair<Eigen::Vector4f, Eigen::Vector3f>
OptimTransformation(const std::vector<Eigen::Vector3f> &src,
                    const std::vector<Eigen::Vector3f> &dst, Eigen::Vector4f r,
                    Eigen::Vector3f t, size_t max_iter) {
  assert(src.size() == dst.size());

  auto linearSolver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto blockSolver =
      std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  g2o::SparseOptimizer *optimizer = new g2o::SparseOptimizer();
  optimizer->setAlgorithm(solver);

  g2o::ParameterSE3Offset *parameter_world_offset =
      new g2o::ParameterSE3Offset();
  parameter_world_offset->setId(0);
  optimizer->addParameter(parameter_world_offset);

  g2o::VertexSE3 *v_se3_base = new g2o::VertexSE3();
  Eigen::Matrix3d R_base = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_base = Eigen::Vector3d::Zero();
  v_se3_base->setEstimate(g2o::SE3Quat(R_base, t_base));
  v_se3_base->setId(0);
  v_se3_base->setFixed(true);
  optimizer->addVertex(v_se3_base);

  g2o::VertexSE3 *v_se3 = new g2o::VertexSE3();
  Eigen::Matrix3d R = Eigen::Quaterniond(r.cast<double>()).toRotationMatrix();
  v_se3->setEstimate(g2o::SE3Quat(R, t.cast<double>()));
  v_se3->setId(1);
  optimizer->addVertex(v_se3);

  for (size_t i = 0; i < src.size(); ++i) {
    g2o::VertexPointXYZ *v_p = new g2o::VertexPointXYZ();
    v_p->setEstimate(src[i].cast<double>());
    v_p->setId(i + 2);
    v_p->setFixed(true);
    optimizer->addVertex(v_p);

    auto *e_base = new g2o::EdgeSE3PointXYZ();
    e_base->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                             optimizer->vertex(i + 2)));
    e_base->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer->vertex(0)));
    e_base->setMeasurement(src[i].cast<double>());
    // e_base->setInformation(Eigen::Matrix3d::Identity());
    e_base->setParameterId(0, 0);

    optimizer->addEdge(e_base);

    auto *e = new g2o::EdgeSE3PointXYZ();
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer->vertex(i + 2)));
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer->vertex(1)));
    e->setMeasurement(dst[i].cast<double>());
    // e->setInformation(Eigen::Matrix3d::Identity());
    e->setParameterId(0, 0);

    optimizer->addEdge(e);
  }

  optimizer->initializeOptimization();

  auto start = std::chrono::high_resolution_clock::now();
  optimizer->optimize(max_iter);

  R = dynamic_cast<g2o::VertexSE3 *>(optimizer->vertex(1))
          ->estimate()
          .rotation();
  r = Eigen::Quaternionf(R.cast<float>()).coeffs();
  t = dynamic_cast<g2o::VertexSE3 *>(optimizer->vertex(1))
          ->estimate()
          .translation()
          .cast<float>();

  delete optimizer;

  return {r, t};
}

void LocalBundleAdjustment(KeyFrame::Ptr ckf, Map::Ptr map) {
  if (ckf->get_bad() || !ckf->get_on_map()) {
    return;
  }
  std::unordered_map<uuids::uuid, KeyFrame::Ptr> neighbor_kfs_map;
  for (auto mp : ckf->get_mps()) {
    if (mp == nullptr || mp->get_bad() || !mp->get_on_map()) {
      continue;
    }
    for (auto &obs : mp->get_observations()) {
      auto [kf_weak, idx] = obs.second;
      if (auto kf = kf_weak.lock()) {
        if (!kf->get_bad() && kf->get_on_map()) {
          neighbor_kfs_map[kf->get_uuid()] = kf;
        }
      }
    }
  }
  if (neighbor_kfs_map.find(ckf->get_uuid()) != neighbor_kfs_map.end()) {
    neighbor_kfs_map.erase(ckf->get_uuid());
  }
  if (neighbor_kfs_map.empty()) {
    return;
  }

  std::vector<KeyFrame::Ptr> local_kfs;
  local_kfs.reserve(neighbor_kfs_map.size());
  for (auto item : neighbor_kfs_map) {
    local_kfs.push_back(item.second);
  }
  local_kfs.push_back(ckf);
  if (local_kfs.size() < 4) {
    return;
  }

  std::vector<MapPoint<KeyFrame>::Ptr> local_mps;
  std::unordered_map<uuids::uuid, MapPoint<KeyFrame>::Ptr> local_mps_map;
  for (auto kf : local_kfs) {
    auto mps = kf->get_mps();
    for (auto mp : mps) {
      if (mp == nullptr || mp->get_bad() || !mp->get_on_map()) {
        continue;
      }
      local_mps_map[mp->get_uuid()] = mp;
    }
  }
  for (auto item : local_mps_map) {
    local_mps.push_back(item.second);
  }
  if (local_mps.empty() || local_mps.size() < 4) {
    return;
  }

  std::unordered_map<uuids::uuid, KeyFrame::Ptr> fixed_kfs_map;
  for (auto mp : local_mps) {
    for (auto &obs : mp->get_observations()) {
      if (auto kf = obs.second.first.lock()) {
        if (!kf->get_bad() && kf->get_on_map()) {
          fixed_kfs_map[kf->get_uuid()] = kf;
        }
      }
    }
  }

  for (auto kf : local_kfs) {
    if (fixed_kfs_map.find(kf->get_uuid()) != fixed_kfs_map.end()) {
      fixed_kfs_map.erase(kf->get_uuid());
    }
  }

  std::vector<KeyFrame::Ptr> fixed_kfs;
  fixed_kfs.reserve(fixed_kfs_map.size());
  for (auto item : fixed_kfs_map) {
    fixed_kfs.push_back(item.second);
  }

  std::unordered_map<uuids::uuid, size_t> kf_to_idx;
  for (size_t i = 0; i < local_kfs.size(); ++i) {
    kf_to_idx[local_kfs[i]->get_uuid()] = i;
  }
  for (size_t i = 0; i < fixed_kfs.size(); ++i) {
    kf_to_idx[fixed_kfs[i]->get_uuid()] = i + local_kfs.size();
  }
  // 最早的关键帧
  time_t min_time = std::numeric_limits<time_t>::max();
  size_t earliest_kf_idx = 0;
  for (size_t i = 0; i < local_kfs.size(); ++i) {
    if (local_kfs[i]->get_timestamp() < min_time) {
      min_time = local_kfs[i]->get_timestamp();
      earliest_kf_idx = kf_to_idx[local_kfs[i]->get_uuid()];
    }
  }

  auto linearSolver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto blockSolver =
      std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  g2o::SparseOptimizer *optimizer = new g2o::SparseOptimizer();
  optimizer->setAlgorithm(solver);

  g2o::ParameterSE3Offset *parameter_world_offset =
      new g2o::ParameterSE3Offset();
  parameter_world_offset->setId(0);
  optimizer->addParameter(parameter_world_offset);

  for (size_t i = 0; i < local_kfs.size(); ++i) {
    g2o::VertexSE3 *v_se3 = new g2o::VertexSE3();
    Eigen::Quaterniond r_quat(local_kfs[i]->get_r_cw().cast<double>());
    Eigen::Matrix3d R = r_quat.toRotationMatrix();
    Eigen::Vector3d t = local_kfs[i]->get_t_cw().cast<double>();
    v_se3->setEstimate(g2o::SE3Quat(R, t));
    v_se3->setId(kf_to_idx[local_kfs[i]->get_uuid()]);
    if (local_kfs[i]->get_fixed() ||
        local_kfs[i]->get_parents_kf() == nullptr ||
        kf_to_idx[local_kfs[i]->get_uuid()] == earliest_kf_idx) {
      v_se3->setFixed(true);
    } else {
      v_se3->setFixed(false);
    }

    optimizer->addVertex(v_se3);
  }
  for (size_t i = 0; i < fixed_kfs.size(); ++i) {
    g2o::VertexSE3 *v_se3 = new g2o::VertexSE3();
    Eigen::Quaterniond r_quat(fixed_kfs[i]->get_r_cw().cast<double>());
    Eigen::Matrix3d R = r_quat.toRotationMatrix();
    Eigen::Vector3d t = fixed_kfs[i]->get_t_cw().cast<double>();
    v_se3->setEstimate(g2o::SE3Quat(R, t));
    v_se3->setId(kf_to_idx[fixed_kfs[i]->get_uuid()]);
    v_se3->setFixed(true);
    optimizer->addVertex(v_se3);
  }
  for (size_t i = 0; i < local_kfs.size(); ++i) {
    auto pkf = local_kfs[i]->get_parents_kf();
    if (pkf != nullptr) {
      if (kf_to_idx.find(pkf->get_uuid()) != kf_to_idx.end()) {
        auto *e = new g2o::EdgeSE3();
        e->setVertex(
            0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                   optimizer->vertex(kf_to_idx[local_kfs[i]->get_uuid()])));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer->vertex(kf_to_idx[pkf->get_uuid()])));
        Eigen::Quaterniond qp(pkf->get_r_cw().cast<double>());
        Eigen::Quaterniond ql(local_kfs[i]->get_r_cw().cast<double>());
        Eigen::Quaterniond q = qp.conjugate() * ql;
        Eigen::Matrix3d R = q.toRotationMatrix();
        auto t = local_kfs[i]->get_t_cw().cast<double>() -
                 pkf->get_t_cw().cast<double>();
        e->setMeasurement(g2o::SE3Quat(R, t));
        // e->setInformation(Eigen::Matrix3d::Identity());
        optimizer->addEdge(e);
      }
    }
  }

  for (size_t i = 0; i < local_mps.size(); ++i) {
    auto id_mp = i + kf_to_idx.size();
    g2o::VertexPointXYZ *v_p = new g2o::VertexPointXYZ();
    v_p->setEstimate(local_mps[i]->get_pos().cast<double>());
    v_p->setId(id_mp);

    optimizer->addVertex(v_p);

    for (auto &obs : local_mps[i]->get_observations()) {
      auto [kf_weak, m_idx] = obs.second;
      if (auto kf = kf_weak.lock()) {
        auto kf_idx = kf_to_idx.find(kf->get_uuid());
        if (kf_idx == kf_to_idx.end()) {
          continue;
        }

        Eigen::Vector3d measurement =
            kf->get_measurement()[m_idx].position.cast<double>();

        auto *e = new g2o::EdgeSE3PointXYZ();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer->vertex(kf_idx->second)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer->vertex(id_mp)));

        e->setMeasurement(measurement);
        e->setParameterId(0, 0);
        auto robust_kernel = new g2o::RobustKernelHuber;
        robust_kernel->setDelta(5.99);
        e->setRobustKernel(robust_kernel);

        Eigen::Matrix3f cov =
            GetPoint3dCov(measurement.cast<float>(), kf->get_camera()->K(0, 0),
                          kf->get_camera()->K(1, 1), kf->get_camera()->K(0, 2),
                          kf->get_camera()->K(1, 2));

        e->setInformation(cov.inverse().cast<double>());

        optimizer->addEdge(e);
      }
    }
  }

  optimizer->initializeOptimization();
  auto start = std::chrono::high_resolution_clock::now();
  optimizer->optimize(5);

  // TODO 移除误差较大的点和关键帧
  // 更新关键帧位姿
  for (size_t i = 0; i < local_kfs.size(); ++i) {
    auto v_se3 = dynamic_cast<g2o::VertexSE3 *>(
        optimizer->vertex(kf_to_idx[local_kfs[i]->get_uuid()]));
    auto se3 = v_se3->estimate();
    Eigen::Matrix3f R = se3.rotation().cast<float>();

    local_kfs[i]->set_r_cw(R);
    local_kfs[i]->set_t_cw(se3.translation().cast<float>());
  }
  // 更新地图点位置
  for (size_t i = 0; i < local_mps.size(); ++i) {
    auto v_p = dynamic_cast<g2o::VertexPointXYZ *>(
        optimizer->vertex(i + kf_to_idx.size()));
    // ARGS
    if ((local_mps[i]->get_pos() - v_p->estimate().cast<float>()).norm() <
        0.3) {
      local_mps[i]->set_pos(v_p->estimate().cast<float>());
    } else {
      local_mps[i]->set_bad(true);
    }
  }
  delete optimizer;
}


void OptimizeGBSubMap(GBSubMap *submap,  uuids::uuid fix_kf,
                                          size_t max_iter) {
  if (!submap) {
    return;
  }

  // 创建优化器
  auto linearSolver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto blockSolver =
      std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(false);

  // 映射uuid到顶点id
  std::unordered_map<uuids::uuid, size_t> kf_id_map;
  size_t vertex_id = 0;

  // 收集所有地图的关键帧
  const auto &local_maps = submap->get_local_maps();

  // 遍历所有子地图的关键帧
  for (const auto &[map_uuid, map] : local_maps) {
    const auto &kfs = map->get_kfs();
    for (const auto &[kf_uuid, kf] : kfs) {
      if (kf->get_gb_init()) {
        g2o::VertexSE3 *v_se3 = new g2o::VertexSE3();
        Eigen::Quaterniond r_quat(kf->get_gb_r_cw().cast<double>());
        Eigen::Matrix3d R = r_quat.toRotationMatrix();
        Eigen::Vector3d t = kf->get_gb_t_cw().cast<double>();
        v_se3->setEstimate(g2o::SE3Quat(R, t));
        v_se3->setId(vertex_id);

        kf_id_map[kf_uuid] = vertex_id++;
        optimizer.addVertex(v_se3);
      }
    }
  }

  // 固定第一个关键帧位姿
  if (!kf_id_map.empty() && !fix_kf.is_nil() && kf_id_map.find(fix_kf) != kf_id_map.end()) {
    auto fix_vertex = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(kf_id_map[fix_kf]));
    if (fix_vertex) {
      fix_vertex->setFixed(true);
    }
  }

  // 添加父子关键帧之间的约束边
  for (const auto &[map_uuid, map] : local_maps) {
    const auto &kfs = map->get_kfs();
    // AddParentChildEdges(&optimizer, kfs, kf_id_map);
    for (const auto &[kf_uuid, kf] : kfs) {
      auto it_curr = kf_id_map.find(kf_uuid);
      if (it_curr == kf_id_map.end()) {
        continue;
      }

      // 父子关键帧约束
      auto pre_kf_uuid = kf->get_pre_kf();
      if (!pre_kf_uuid.is_nil()) {
        auto it_pre = kf_id_map.find(pre_kf_uuid);
        if (it_pre != kf_id_map.end()) {
          auto *edge = new g2o::EdgeSE3();
          edge->setVertex(0, optimizer.vertex(it_curr->second));
          edge->setVertex(1, optimizer.vertex(it_pre->second));

          // 计算相对位姿约束
          Eigen::Quaternionf rel_r(kf->get_rel_r_cw());
          Eigen::Vector3f rel_t = kf->get_rel_t_cw();

          Eigen::Matrix3d R = rel_r.toRotationMatrix().cast<double>();
          Eigen::Vector3d t = rel_t.cast<double>();

          edge->setMeasurement(g2o::SE3Quat(R, t));

          // 设置信息矩阵（协方差的逆）
          Eigen::Matrix<double, 6, 6> information =
              Eigen::Matrix<double, 6, 6>::Identity();
          edge->setInformation(information);

          // 添加鲁棒核函数
          g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
          rk->setDelta(1.0);
          edge->setRobustKernel(rk);

          optimizer.addEdge(edge);
        }
      }
    }
  }

  // 添加跨地图关键帧之间的约束边
  // AddCrossMapEdges(&optimizer, submap, kf_id_map);
  const auto &reloc_info = submap->get_reloc_info();

  for (const auto &[reloc_uuid, info] : reloc_info) {
    auto &[map_pair, constraint] = info;
    auto &[map_uuid_1, map_uuid_2] = map_pair;
    auto &[kf_pair, transform] = constraint;
    auto &[kf_uuid_1, kf_uuid_2] = kf_pair;
    auto &[reloc_r, reloc_t] = transform;

    // 查找关键帧顶点ID
    auto it_kf1 = kf_id_map.find(kf_uuid_1);
    auto it_kf2 = kf_id_map.find(kf_uuid_2);

    if (it_kf1 != kf_id_map.end() && it_kf2 != kf_id_map.end()) {
      // 添加跨地图约束边
      auto *edge = new g2o::EdgeSE3();
      edge->setVertex(0, optimizer.vertex(it_kf1->second));
      edge->setVertex(1, optimizer.vertex(it_kf2->second));

      // 设置相对位姿约束
      Eigen::Quaternionf rel_r(reloc_r);
      Eigen::Vector3f rel_t = reloc_t;

      Eigen::Matrix3d R = rel_r.toRotationMatrix().cast<double>();
      Eigen::Vector3d t = rel_t.cast<double>();

      edge->setMeasurement(g2o::SE3Quat(R, t));

      // 设置信息矩阵（跨地图约束可能权重较低）
      Eigen::Matrix<double, 6, 6> information =
          Eigen::Matrix<double, 6, 6>::Identity() * 0.8;
      edge->setInformation(information);

      // 添加鲁棒核函数
      g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
      rk->setDelta(2.0);
      edge->setRobustKernel(rk);

      optimizer.addEdge(edge);
    }
  }

  // 执行优化
  optimizer.initializeOptimization();
  optimizer.optimize(max_iter);

  // 更新关键帧位姿
  for (const auto &[map_uuid, map] : local_maps) {
    for (auto &[kf_uuid, kf] : map->get_kfs()) {
      auto it = kf_id_map.find(kf_uuid);
      if (it != kf_id_map.end()) {
        auto vertex =
            dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(it->second));
        if (vertex) {
          auto se3 = vertex->estimate();
          Eigen::Quaternionf q(se3.rotation().cast<float>());
          kf->set_gb_r_cw(q.coeffs());
          kf->set_gb_t_cw(se3.translation().cast<float>());
        }
      }
    }
  }
}

void GlobalBundleAdjustment(GBSubMap *submap, uuids::uuid fix_kf,size_t max_iter){
  if (!submap) {
    return;
  }

  // 创建优化器
  auto linearSolver = std::make_unique<
      g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto blockSolver =
      std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(false);

  // 关键帧和路标点的UUID到顶点ID的映射
  std::unordered_map<uuids::uuid, size_t> kf_id_map;
  std::unordered_map<uuids::uuid, size_t> mp_id_map;
  size_t vertex_id = 0;

  // 收集所有地图的关键帧和路标点
  const auto &local_maps = submap->get_local_maps();

  // 1. 添加关键帧顶点
  for (const auto &[map_uuid, map] : local_maps) {
    const auto &kfs = map->get_kfs();
    // AddKeyFrameVertices(&optimizer, kfs, kf_id_map, vertex_id);
    for (const auto &[kf_uuid, kf] : kfs) {
      if (kf->get_gb_init()) {
        g2o::VertexSE3 *v_se3 = new g2o::VertexSE3();
        Eigen::Quaterniond r_quat(kf->get_gb_r_cw().cast<double>());
        Eigen::Matrix3d R = r_quat.toRotationMatrix();
        Eigen::Vector3d t = kf->get_gb_t_cw().cast<double>();
        v_se3->setEstimate(g2o::SE3Quat(R, t));
        v_se3->setId(vertex_id);

        kf_id_map[kf_uuid] = vertex_id++;
        optimizer.addVertex(v_se3);
      }
    }
  }

  // 2. 固定指定的关键帧位姿（如果存在）


  // 如果没有指定固定关键帧或指定的关键帧不存在，则固定第一个关键帧
  if (!kf_id_map.empty() && !fix_kf.is_nil() &&
      kf_id_map.find(fix_kf) != kf_id_map.end()) {
    auto fix_vertex =
        dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(kf_id_map[fix_kf]));
    if (fix_vertex) {
      fix_vertex->setFixed(true);
    }
  }

  // 3. 添加路标点顶点
  for (const auto &[map_uuid, map] : local_maps) {
    const auto &mps = map->get_mps();
    for (const auto &[mp_uuid, mp] : mps) {
      if (mp->get_gb_init() && !mp->get_observations().empty()) {
        g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
        v_point->setEstimate(mp->get_gb_pos().cast<double>());
        v_point->setId(vertex_id);
        v_point->setMarginalized(true);  // 边缘化加速求解

        mp_id_map[mp_uuid] = vertex_id++;
        optimizer.addVertex(v_point);
      }
    }
  }

  // 4. 添加投影边（关键帧到路标点的观测）
  for (const auto &[map_uuid, map] : local_maps) {
    for (const auto &[kf_uuid, kf] : map->get_kfs()) {
      // 跳过未添加到优化器的关键帧
      auto kf_it = kf_id_map.find(kf_uuid);
      if (kf_it == kf_id_map.end()) {
        continue;
      }

      // 获取关键帧的观测信息
      const auto &observations = kf->get_measurement_mps();

      for (const auto &[mp_uuid, mp_pos_cam] : observations) {
        // 跳过未添加到优化器的路标点
        auto mp_it = mp_id_map.find(mp_uuid);
        if (mp_it == mp_id_map.end()) {
          continue;
        }

        // 创建投影边
        g2o::EdgeSE3PointXYZ *edge = new g2o::EdgeSE3PointXYZ();
        edge->setVertex(0, optimizer.vertex(kf_it->second));  // 关键帧顶点
        edge->setVertex(1, optimizer.vertex(mp_it->second));  // 路标点顶点

        // 设置测量值（路标点在相机坐标系下的坐标）
        edge->setMeasurement(mp_pos_cam.cast<double>());

        // // 计算路标点的协方差并设置信息矩阵
        // Eigen::Matrix3f point_cov = GetPoint3dCov(mp_pos_cam, fx, fy, cx,
        // cy); Eigen::Matrix3d information =
        // point_cov.cast<double>().inverse();
        // edge->setInformation(information);

        // 添加鲁棒核函数
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(5.99);
        edge->setRobustKernel(rk);

        // 添加边到优化器
        optimizer.addEdge(edge);
      }
    }
  }
  // 5. 执行优化
  optimizer.initializeOptimization();
  optimizer.optimize(max_iter);

  // 6. 更新关键帧位姿
  for (const auto &[map_uuid, map] : local_maps) {
    for (auto &[kf_uuid, kf] : map->get_kfs()) {
      auto it = kf_id_map.find(kf_uuid);
      if (it != kf_id_map.end()) {
        auto vertex =
            dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(it->second));
        if (vertex) {
          auto se3 = vertex->estimate();
          Eigen::Quaternionf q(se3.rotation().cast<float>());
          kf->set_gb_r_cw(q.coeffs());
          kf->set_gb_t_cw(se3.translation().cast<float>());
          kf->set_gb_init(true);
        }
      }
    }
  }

  // 7. 更新路标点位置
  for (const auto &[map_uuid, map] : local_maps) {
    for (auto &[mp_uuid, mp] : map->get_mps()) {
      auto it = mp_id_map.find(mp_uuid);
      if (it != mp_id_map.end()) {
        auto vertex =
            dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(it->second));
        if (vertex) {
          mp->set_gb_pos(vertex->estimate().cast<float>());
          mp->set_gb_init(true);
        }
      }
    }
  }
}

} // namespace rgor