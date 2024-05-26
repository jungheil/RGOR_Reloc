#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#pragma once
#ifndef XI_DRAW_SPHERE_H
#define XI_DRAW_SPHERE_H

void createSphereMesh(std::vector<cv::Point3f>& vertices,
                      std::vector<std::vector<int>>& indices, float radius,
                      int num_layers) {
  // 定义球面上的点
  for (int i = 0; i <= num_layers; ++i) {
    float theta = i * M_PI / num_layers;  // 纬度
    for (int j = 0; j <= num_layers * 2; ++j) {
      float phi = j * 2 * M_PI / (num_layers * 2);  // 经度
      float x = radius * sin(theta) * cos(phi);
      float y = radius * sin(theta) * sin(phi);
      float z = radius * cos(theta);
      vertices.push_back(cv::Point3f(x, y, z));
    }
  }

  // 创建三角网格
  for (int i = 0; i < num_layers; ++i) {
    for (int j = 0; j < num_layers * 2; ++j) {
      int next_i = i + 1;
      int next_j = (j + 1) % (num_layers * 2 + 1);
      indices.push_back({i * (num_layers * 2 + 1) + j,
                         next_i * (num_layers * 2 + 1) + j,
                         i * (num_layers * 2 + 1) + next_j});
      indices.push_back({next_i * (num_layers * 2 + 1) + j,
                         next_i * (num_layers * 2 + 1) + next_j,
                         i * (num_layers * 2 + 1) + next_j});
    }
  }
}

void projectPointsOntoImage(const std::vector<cv::Point3f>& objectPoints,
                            const cv::Mat& intrinsics, const cv::Mat& rvec,
                            const cv::Mat& tvec,
                            std::vector<cv::Point2f>& imagePoints) {
  cv::projectPoints(objectPoints, rvec, tvec, intrinsics, cv::Mat(),
                    imagePoints);
}

void drawSphereMesh(cv::Mat& src, cv::Point3f center, float radius, cv::Mat K,
                    cv::Scalar color, int thickness, size_t layers = 6) {
  std::vector<cv::Point3f> vertices;
  std::vector<std::vector<int>> indices;
  createSphereMesh(vertices, indices, radius, layers);

  cv::Mat rvec = (cv::Mat_<double>(3, 1) << 3.14 / 2, 0, 0);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << center.x, center.y, center.z);

  std::vector<cv::Point2f> imagePoints;
  projectPointsOntoImage(vertices, K, rvec, tvec, imagePoints);

  for (const auto& triangle : indices) {
    std::vector<cv::Point> trianglePoints;
    for (int idx : triangle) {
      if (idx < imagePoints.size()) {
        trianglePoints.push_back(imagePoints[idx]);
      } else {
        std::cerr << "Index out of bounds: " << idx << std::endl;
        return;
      }
    }
    if (trianglePoints.size() == 3) {
      std::vector<std::vector<cv::Point>> triangleContours;
      triangleContours.push_back(trianglePoints);
      cv::polylines(src, triangleContours, true, color, thickness);
    } else {
      std::cerr << "Triangle does not have three points." << std::endl;
    }
  }
}

#endif // XI_DRAW_SPHERE_H