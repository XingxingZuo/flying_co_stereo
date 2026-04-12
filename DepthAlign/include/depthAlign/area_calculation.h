//
// Created by wzy on 25-4-26.
//

#ifndef RELATIVE_POSE_BACKEND_AREA_CALCULATION_H
#define RELATIVE_POSE_BACKEND_AREA_CALCULATION_H
#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <flann/flann.hpp>  // 需要FLANN库支持
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <Eigen/Dense>
#include <algorithm>

struct GridHash {
    size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
    }
};

// compute XY plane voxel_size
double computeXYGridCoverageArea(const std::vector<Eigen::Vector3d>& point_cloud, double voxel_size = 0.1) {
  std::unordered_set<std::pair<int, int>, GridHash> grid_cells;

  for (const auto& pt : point_cloud) {
    int x_idx = static_cast<int>(std::floor(pt.x() / voxel_size));
    int y_idx = static_cast<int>(std::floor(pt.y() / voxel_size));
    grid_cells.emplace(x_idx, y_idx);
  }

  double area = grid_cells.size() * voxel_size * voxel_size;
  std::cout << "[computeXYGridCoverageArea] Unique XY grid cells: " << grid_cells.size() << ", total area: " << area << " m^2" << std::endl;
  return area;
}


struct Point2D {
    double x, y;
    bool operator<(const Point2D& p) const {
      return x < p.x || (x == p.x && y < p.y);
    }
};

// Cross product of (p1 -> p2) x (p1 -> p3)
double cross(const Point2D& p1, const Point2D& p2, const Point2D& p3) {
  return (p2.x - p1.x) * (p3.y - p1.y) -
         (p2.y - p1.y) * (p3.x - p1.x);
}

// Compute area of a polygon
double polygonArea(const std::vector<Point2D>& poly) {
  double area = 0.0;
  int n = poly.size();
  for (int i = 0; i < n; ++i) {
    const Point2D& p1 = poly[i];
    const Point2D& p2 = poly[(i + 1) % n];
    area += (p1.x * p2.y - p2.x * p1.y);
  }
  return std::abs(area) * 0.5;
}

// Compute convex hull and return the hull points
std::vector<Point2D> computeConvexHull(const std::vector<Eigen::Vector3d>& point_cloud) {
  std::vector<Point2D> points;
  for (const auto& p : point_cloud) {
    points.push_back({p.x(), p.y()});
  }

  if (points.size() < 3) {
    return {};
  }

  // Sort points
  std::sort(points.begin(), points.end());

  // Build lower and upper hull
  std::vector<Point2D> hull;
  for (const auto& p : points) {
    while (hull.size() >= 2 && cross(hull[hull.size()-2], hull[hull.size()-1], p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  size_t lower_size = hull.size();
  for (int i = points.size() - 2; i >= 0; --i) {
    const auto& p = points[i];
    while (hull.size() > lower_size && cross(hull[hull.size()-2], hull[hull.size()-1], p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  if (!hull.empty()) {
    hull.pop_back(); // Remove duplicate
  }

  return hull;
}

// Draw points and convex hull using OpenCV
void visualizeConvexHull(const std::vector<Eigen::Vector3d>& point_cloud, const std::vector<Point2D>& hull) {
  // 1. Find bounds
  double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
  for (const auto& p : point_cloud) {
    min_x = std::min(min_x, p.x());
    max_x = std::max(max_x, p.x());
    min_y = std::min(min_y, p.y());
    max_y = std::max(max_y, p.y());
  }

  double padding = 20.0; // pixels
  double scale = 500.0 / std::max(max_x - min_x, max_y - min_y); // roughly fit 500px size
  int img_w = static_cast<int>((max_x - min_x) * scale + 2 * padding);
  int img_h = static_cast<int>((max_y - min_y) * scale + 2 * padding);

  cv::Mat img(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));

  // 2. Draw all points
  for (const auto& p : point_cloud) {
    int u = static_cast<int>((p.x() - min_x) * scale + padding);
    int v = static_cast<int>((max_y - p.y()) * scale + padding); // flip y for image
    cv::circle(img, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), -1); // red points
  }

  // 3. Draw convex hull
  for (size_t i = 0; i < hull.size(); ++i) {
    const auto& p1 = hull[i];
    const auto& p2 = hull[(i + 1) % hull.size()];
    int u1 = static_cast<int>((p1.x - min_x) * scale + padding);
    int v1 = static_cast<int>((max_y - p1.y) * scale + padding);
    int u2 = static_cast<int>((p2.x - min_x) * scale + padding);
    int v2 = static_cast<int>((max_y - p2.y) * scale + padding);
    cv::line(img, cv::Point(u1, v1), cv::Point(u2, v2), cv::Scalar(0, 255, 0), 2); // green lines
  }

  // 4. Show
  cv::imshow("Convex Hull", img);
  cv::waitKey(0);
  cv::destroyAllWindows();
}


#endif //RELATIVE_POSE_BACKEND_AREA_CALCULATION_H
