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

#ifndef RGOR_OPTIMIZER_H
#define RGOR_OPTIMIZER_H

#include "common/Frame.h"
#include "common/Map.h"

namespace rgor {
std::pair<Eigen::Vector4f, Eigen::Vector3f>
OptimTransformation(const std::vector<Eigen::Vector3f> &src,
                    const std::vector<Eigen::Vector3f> &dst, Eigen::Vector4f r,
                    Eigen::Vector3f t, size_t max_iter = 10);

void LocalBundleAdjustment(KeyFrame::Ptr kf, Map::Ptr map);
} // namespace rgor

#endif // RGOR_OPTIMIZER_H
