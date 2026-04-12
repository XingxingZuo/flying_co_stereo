//
// Created by wzy on 25-2-24.
//
#ifndef RELATIVE_POSE_BACKEND_DEPTHALIGNCOMMON_H
#define RELATIVE_POSE_BACKEND_DEPTHALIGNCOMMON_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>
#include "ceres/ceres.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

struct ExponentialResidual {
    ExponentialResidual(double x, double y, double offset, double c)
            : x_(x), y_(y), offset_(offset), c_(c) {}

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
      residual[0] = y_ - (a[0] * ceres::exp(b[0] * (x_ - offset_)) + c_);
      return true;
    }

private:
    const double x_, y_, offset_, c_;
};

cv::Mat plotXYDataExp(std::vector<double> &x, std::vector<double> &y, std::tuple<double, double, double, double> &exp_params_opti) {
  plt::figure_size(640, 480);
  plt::scatter(x, y, 10.0, {{"color", "green"}, {"label", "Landmark Points"}});
  double x_min = *std::min_element(x.begin(), x.end());
  double x_max = *std::max_element(x.begin(), x.end());
  std::vector<double> x_array;
  for (double value = x_min; value <= x_max; value += 2.0) {
    x_array.push_back(value);
  }

  auto [a, b, offset, c] = exp_params_opti;
  std::vector<double> exp_fit_y;
  for (const auto &xi : x_array) {
    exp_fit_y.push_back(a * std::exp(b * (xi - offset)) + c);
  }

  if (!exp_fit_y.empty()) {
    plt::plot(x_array, exp_fit_y, {{"label", "Exponential Fit Opti"}, {"color", "m"}});
  } else {
    plt::text(0.5, 0.5, "Exponential fit failed");
  }

  double text_x_min = *std::min_element(x_array.begin(), x_array.end());
  double text_x_max = *std::max_element(x_array.begin(), x_array.end());
  double text_y_min = *std::min_element(exp_fit_y.begin(), exp_fit_y.end());
  double text_y_max = *std::max_element(exp_fit_y.begin(), exp_fit_y.end());

  double text_x = text_x_min + 0.1 * (x_max - text_x_min);
  double text_y = text_y_min + 0.8 * (text_y_max - text_y_min);
  std::cout << "======== text_x: " << text_x << std::endl;
  std::cout << "======== text_y: " << text_y << std::endl;
  std::string fit_equation = "y = " +
                             std::to_string(a) + "e^(" +
                             std::to_string(b) + "*(x-" +
                             std::to_string(offset) + ")) + " +
                             std::to_string(c);
//  plt::text(0.02, 0.8, fit_equation);
//  plt::text(text_x, text_y,  fit_equation);

  plt::title("Depth Fit", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
  plt::xlabel("Predict Relative Depth (grayscale)", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
  plt::ylabel("Measured True Depth (m)", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
//  plt::ylim(0, 80);
  plt::legend();
  plt::grid(true);
//  plt::show();
  std::string tmp_filename = "/tmp/plot.png";
  plt::save(tmp_filename);
  plt::close();
  return cv::imread(tmp_filename);
}

// exponential fit to find the parameters a, b, offset and c for the function y = a * exp(b * (x - offset)) + c
std::tuple<std::vector<double>, std::vector<double>, double, double, double, double> exponential_fit_params(
        const std::vector<double>& x,
        const std::vector<double>& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("Input vectors x and y must have the same size.");
  }

  std::vector<std::pair<double, double>> xy_pairs;
  for (size_t i = 0; i < x.size(); ++i) {
    xy_pairs.emplace_back(x[i], y[i]);
  }
  std::sort(xy_pairs.begin(), xy_pairs.end());
  std::vector<double> sorted_x, sorted_y;
  for (const auto& pair : xy_pairs) {
    sorted_x.push_back(pair.first);
    sorted_y.push_back(pair.second);
  }

  // 2. calculate the minimum and maximum of x and y to determine the range for fitting
  double x_min = sorted_x.front();
  double x_max = sorted_x.back();
  double y_min = *std::min_element(sorted_y.begin(), sorted_y.end());
  printf("x_min = %.3f, x_max = %.3f, y_min = %.3f\n", x_min, x_max, y_min);

  // 3. set initial values for a, b, offset and c. We can set offset to x_min, and c to y_min - 1.0 to ensure the curve starts below the minimum y value, and then optimize a and b to fit the data.
  double a,b;
  double offset = x_min;
  double c = y_min - 1.0;

  // 4. use Ceres Solver to optimize the parameters a, b, offset and c to fit the data
  ceres::Problem problem;
  for (size_t i = 0; i < sorted_x.size(); ++i) {
    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                    new ExponentialResidual(sorted_x[i], sorted_y[i], offset, c)),
            nullptr, &a, &b);
  }

  // 5. set options for the solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;

  // 6. solve the optimization problem
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  std::cout << "Optimized a: " << a << ", b: " << b << ", offset: " <<  offset << ", c: " << c << std::endl;

  std::tuple<double, double, double, double> exp_params_opti = {a, b, offset, c};
  plotXYDataExp(sorted_x, sorted_y, exp_params_opti);

  return {sorted_x, sorted_y, a, b, offset, c};
}


cv::Mat plotXYDataLinear(std::vector<double> &x, std::vector<double> &y, std::tuple<double, double> &linear_params_opti) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("[plotXYDataLinear] Input vectors x and y must have the same size. x.size() = " + std::to_string(x.size()) + ", y.size() = " + std::to_string(y.size()));
  }
  plt::figure_size(640, 480);
  plt::scatter(x, y, 10.0, {{"color", "green"}, {"label", "Landmark Points"}});
  double x_min = *std::min_element(x.begin(), x.end());
  double x_max = *std::max_element(x.begin(), x.end());
  std::vector<double> x_array;
  for (double value = x_min; value <= x_max; value += 2.0) {
    x_array.push_back(value);
  }
  auto [a, b] = linear_params_opti;
  std::vector<double> linear_fit_y;
  for (const auto &xi : x_array) {
    linear_fit_y.push_back(a * xi + b);
  }
  if (!linear_fit_y.empty()) {
    plt::plot(x_array, linear_fit_y, {{"label", "Linear Fit Opti"}, {"color", "m"}});
  } else {
    plt::text(0.5, 0.5, "Linear fit failed");
  }

  double text_x_min = *std::min_element(x_array.begin(), x_array.end());
  double text_x_max = *std::max_element(x_array.begin(), x_array.end());
  double text_y_min = *std::min_element(linear_fit_y.begin(), linear_fit_y.end());
  double text_y_max = *std::max_element(linear_fit_y.begin(), linear_fit_y.end());
  double text_x = text_x_min + 0.1 * (x_max - text_x_min);
  double text_y = text_y_min + 0.8 * (text_y_max - text_y_min);
  std::cout << "======== text_x: " << text_x << std::endl;
  std::cout << "======== text_y: " << text_y << std::endl;
  std::string fit_equation = "y = " +
                             std::to_string(a) + "*x" +
                             std::to_string(b);

  plt::title("Depth Fit", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
  plt::xlabel("Predict Relative Depth (grayscale)", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
  plt::ylabel("Measured True Depth (m)", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
//  plt::ylim(0, 80);
  plt::legend();
  plt::grid(true);
//  plt::show();

  std::string tmp_filename = "/tmp/plot.png";
  plt::save(tmp_filename);
  plt::close();
  return cv::imread(tmp_filename);
}

// linear fit to find the parameters a and b for the function y = a * x + b, and return the sorted x and y for plotting
std::tuple<std::vector<double>, std::vector<double>> linearFit(const std::vector<double>& x, const std::vector<double>& y, double& a, double& b) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("[linearFit] Input vectors x and y must have the same size.");
  }
  int n = x.size();

  double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
  double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
  double sum_xx = 0.0;
  double sum_xy = 0.0;

  for (int i = 0; i < n; ++i) {
    sum_xx += x[i] * x[i];
    sum_xy += x[i] * y[i];
  }

  a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
  b = (sum_y - a * sum_x) / n;

  std::vector<std::pair<double, double>> xy_pairs;
  for (size_t i = 0; i < x.size(); ++i) {
    xy_pairs.emplace_back(x[i], y[i]);
  }
  std::sort(xy_pairs.begin(), xy_pairs.end());

  std::vector<double> sorted_x, sorted_y;
  for (const auto& pair : xy_pairs) {
    sorted_x.push_back(pair.first);
    sorted_y.push_back(pair.second);
  }
  return {sorted_x, sorted_y};
}

std::tuple<double, double, double> fitQuadraticWithFixedB_C(const std::vector<double>& x, const std::vector<double>& y) {
  int n = x.size();

  double x_min = *std::min_element(x.begin(), x.end());
  double y_min = *std::min_element(y.begin(), y.end());

  Eigen::MatrixXd A(n, 1);
  Eigen::VectorXd B(n);

  for (int i = 0; i < n; i++) {
    A(i, 0) = (x[i] - x_min) * (x[i] - x_min);  // (x - b)^2
    B(i) = y[i] - y_min;  // y - c
  }

  Eigen::VectorXd a = A.colPivHouseholderQr().solve(B);
  std::cout << "y = " << a(0) << "(x - " << x_min << ")^2 + " << y_min << std::endl;
  return {a(0), x_min, y_min};
}

cv::Mat plotXYDataLinearExpQuad(std::vector<double> &x, std::vector<double> &y, std::tuple<double, double> &linear_params,
                                std::tuple<double, double, double, double> &exp_params,
                                std::tuple<double, double, double> &quad_params) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("[plotXYDataLinear] Input vectors x and y must have the same size. x.size() = " + std::to_string(x.size()) + ", y.size() = " + std::to_string(y.size()));
  }
  plt::figure_size(640, 480);
  plt::scatter(x, y, 10.0, {{"color", "green"}, {"label", "Landmark Points"}});

  double x_min = *std::min_element(x.begin(), x.end());
  double x_max = *std::max_element(x.begin(), x.end());
  std::vector<double> x_array;
  for (double value = x_min; value <= x_max; value += 2.0) {
    x_array.push_back(value);
  }

  auto [linear_a, linear_b] = linear_params;
  std::vector<double> linear_fit_y;
  for (const auto &xi : x_array) {
    linear_fit_y.push_back(linear_a * xi + linear_b);
  }
  if (!linear_fit_y.empty()) {
    plt::plot(x_array, linear_fit_y, { {"label", "Linear Fit Opti"}, {"color", "blue"}});
  } else {
    plt::text(0.5, 0.5, "Linear fit failed");
  }

  auto [exp_a, exp_b, exp_c, exp_d] = exp_params;
  std::vector<double> exp_fit_y;
  for (const auto &xi : x_array) {
    exp_fit_y.push_back(exp_a * std::exp(exp_b * (xi - exp_c)) + exp_d);
  }
  if (!exp_fit_y.empty()) {
    plt::plot(x_array, exp_fit_y, { {"label", "Exponential Fit Opti"}, {"color", "red"}});
  } else {
    plt::text(0.5, 0.5, "Exponential fit failed");
  }

  auto [quad_a, quad_b, quad_c] = quad_params;
  std::vector<double> quad_fit_y;
  for (const auto &xi : x_array) {
    quad_fit_y.push_back(quad_a * (xi - quad_b) * (xi - quad_b) + quad_c);
  }
  if (!quad_fit_y.empty()) {
    plt::plot(x_array, quad_fit_y, { {"label", "Quadratic Fit"}, {"color", "purple"}});
  } else {
    plt::text(0.5, 0.5, "Quadratic fit failed");
  }

  plt::title("Depth Fit", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
  plt::xlabel("Predict Relative Depth (grayscale)", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});
  plt::ylabel("Measured True Depth (m)", {{"fontsize", "16"}, {"fontname", "Times New Roman"}});

  plt::legend();
  plt::grid(true);
//  plt::show();

  std::string tmp_filename = "/tmp/plot.png";
  plt::save(tmp_filename);
  plt::close();
  return cv::imread(tmp_filename);
}


bool queryUserIfContinue(){
  std::cout << ">>> Check the above parameter, Do you want to continue? (y/n): ";
  char user_input;
  std::cin >> user_input;

  if (user_input == 'y' || user_input == 'Y') {
    std::cout << "Continue to process !" << std::endl;
  } else {
    std::cout << "Operation canceled by the user." << std::endl;
    exit(-1);
  }
}

#endif //RELATIVE_POSE_BACKEND_DEPTHALIGNCOMMON_H
