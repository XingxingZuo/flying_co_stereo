//
// Created by wzy on 2024/09/01.
//
#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}
#endif
//ros package
#include <ros/ros.h>
#include <geometry_msgs/Point32.h>
#include "sensor_msgs/PointCloud2.h"
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>

///basic C++ packages
#include <vector>
#include <string>
#include <memory>
#include <iostream>

///specific third party packages
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include "matplotlibcpp.h"
#include <depthAlign/depthAlignCommon.h>

namespace plt = matplotlibcpp;

///specific this packages
#include <experimental/filesystem>
#include <regex>
#include <fstream>
namespace fs = std::experimental::filesystem;

using namespace std;
struct Pose{
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    Eigen::Matrix4d Tx = Eigen::Matrix4d::Identity();
};

struct lmd_uv{
    Eigen::Vector2d uv;
    Eigen::Vector3d p_in_w;
};

//global
std::map<double, Pose> time_pose_map;
std::map<double, std::map<size_t, lmd_uv>> time_lmd_map;
int img_width = 640;
int img_height = 480;
cv::Mat K = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0);
#ifdef real_data
cv::Mat K_kun0_camera = (cv::Mat_<float>(3,3) << 381.764, 0.0, 317.666, 0.0, 380.931, 239.637,  0.0, 0.0, 1.0); //kun0 D455 color
cv::Mat K_kun0_infra1 = (cv::Mat_<float>(3,3) << 389.862, 0.0, 315.673, 0.0, 389.862, 240.799,  0.0, 0.0, 1.0); //kun0 D455 infra1
cv::Mat K_iris0_color = (cv::Mat_<float>(3,3) << 320.0, 0.0, 320.5, 0.0, 320.0, 240.5,  0.0, 0.0, 1.0); //iris0 color
int height_max = 30;
size_t ID_base = 4000;
#elif sim_data
cv::Mat K_kun0_camera = (cv::Mat_<float>(3,3) << 319.99882, 0.0, 320.5, 0.0, 319.99882, 240.5,  0.0, 0.0, 1.0); //kun0 D455 color
cv::Mat K_kun0_infra1 = (cv::Mat_<float>(3,3) << 319.99882, 0.0, 320.5, 0.0, 319.99882, 240.5,  0.0, 0.0, 1.0); //kun0 D455 infra1
#endif
Eigen::Matrix3d C_norm2world;
std::string world_frame = "global"; //global
double depth_image_time = 0.0;

std::string root_path, lmd_pose_path, time_img_predict_depth_grayscale_path, time_img_predict_depth_colored_path,
        time_img_raw_path, output_xy_fit_path, output_lmd_pose_path, color_or_infra, output_img_path;

void read_lmd_poses(const std::string &file)
{
  std::vector<std::vector<double>> csv_landmarks;
  FILE *fp;
  fp=fopen(file.c_str(),"r");
  if(!fp)
  {
    ROS_ERROR_STREAM("cannot open file: " + file);
  }
  //skip first line of txt，read from second line
  for (int i=0;i<1;i++) fscanf(fp,"%*[^\n]%*c");

  while(1){
    double timestamp;
    int id;
    double lmd[3];
    double lmd_uv[2];
    double cam_t[3];
    double cam_q[4];
    fscanf(fp,"%lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&timestamp, &id,
           &lmd[0], &lmd[1], &lmd[2], &lmd_uv[0], &lmd_uv[1],
           &cam_t[0], &cam_t[1], &cam_t[2], &cam_q[0], &cam_q[1], &cam_q[2], &cam_q[3]);
    Eigen::Vector3d t(cam_t[0], cam_t[1], cam_t[2]);
    Eigen::Quaterniond q(cam_q[3], cam_q[0], cam_q[1], cam_q[2]);
    time_pose_map[timestamp] = Pose{t, q.normalized()};
    //add filter for abnormal point 0.000
    if(lmd[0] == 0.00 || lmd[1] == 0.00 || lmd[2] == 0.00) continue;
    time_lmd_map[timestamp][id].p_in_w = Eigen::Vector3d(lmd[0], lmd[1], lmd[2]);
    time_lmd_map[timestamp][id].uv = Eigen::Vector2d(lmd_uv[0], lmd_uv[1]);
    if(feof(fp)) break;
  }
  fclose(fp);
}


std::map<double, std::string> read_images_by_timestamp(const std::string& folderPath) {
  std::map<double, std::string> imageMap;
  std::regex timestampRegex(R"((\d+)_([\d]+)\.png)");
  for (const auto& entry : fs::directory_iterator(folderPath)) {
    if (is_regular_file(entry)) {
      std::string fileName = entry.path().filename().string();
      std::smatch match;
      if (std::regex_match(fileName, match, timestampRegex)) {
        std::string integerPart = match[1].str();
        std::string fractionalPart = match[2].str();
        long int integerPartInt = std::stol(integerPart);
        double fractionalValue = std::stod("0." + fractionalPart);
        double timestamp_combine = integerPartInt + fractionalValue;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << timestamp_combine;
        printf("oss = %s\n", oss.str().c_str());

        std::string timestampStr = oss.str();
        double timestamp = std::stod(timestampStr);
        printf("time = %.3f\n", timestamp);
        printf("path = %s\n\n", (folderPath + fileName).c_str());
        imageMap[timestamp] = folderPath + fileName;
      }
    }
  }
  return imageMap;
}

void publishPointCloud(const std::vector<Eigen::Vector3d>& points_in_world, ros::Publisher& pub) {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width = points_in_world.size();
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.points.resize(cloud.width * cloud.height);

  for (size_t i = 0; i < points_in_world.size(); ++i) {
    const Eigen::Vector3d& eigen_point = points_in_world[i];
    pcl::PointXYZ& pcl_point = cloud.points[i];
    pcl_point.x = static_cast<float>(eigen_point.x());
    pcl_point.y = static_cast<float>(eigen_point.y());
    pcl_point.z = static_cast<float>(eigen_point.z());
  }
  sensor_msgs::PointCloud2 ros_cloud;
  pcl::toROSMsg(cloud, ros_cloud);
  ros_cloud.header.frame_id = world_frame;
  ros_cloud.header.stamp = ros::Time(depth_image_time);
  pub.publish(ros_cloud);
}

void publishImages(const std::string& image_path, image_transport::Publisher& image_pub) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
      ROS_ERROR("Failed to load image: %s", image_path.c_str());
      return;
    }
//    cv::imshow("depth_image_raw", img);
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = ros::Time::now();
    cv_img.header.frame_id = "camera_link";
    cv_img.encoding = sensor_msgs::image_encodings::BGR8;
    cv_img.image = img;

    sensor_msgs::ImagePtr ros_img_msg = cv_img.toImageMsg();
    image_pub.publish(ros_img_msg);
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "depth_align_node");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
//    ros::NodeHandle nh("~");
  ros::Rate rate(1);
  // create publishers
  image_transport::Publisher image_pub = it.advertise("camera/depth_image", 1);
  image_transport::Publisher image_raw_pub = it.advertise("camera/raw_image", 1);
  ros::Publisher pointcloud_dense_lmd_pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud_dense", 1);
  ros::Publisher pointcloud_dense_lmd_in_cam_pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud_dense_in_cam", 1);
  ros::Publisher pointcloud_sparse_lmd_pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud_sparse", 1);
  ros::Publisher landmarks_pub = nh.advertise<visualization_msgs::MarkerArray>("/landmark_visual_all_map",1);

  // read param
  nh.param<std::string>("root_path", root_path, "default.txt"); //read pose
  nh.param<std::string>("lmd_pose_path", lmd_pose_path, "default.txt"); //read pose
  nh.param<std::string>("time_img_predict_depth_grayscale_path", time_img_predict_depth_grayscale_path, "default_time_img_path");
  nh.param<std::string>("time_img_predict_depth_colored_path", time_img_predict_depth_colored_path, "default_time_img_colored_depth_path");
  nh.param<std::string>("time_img_raw_path", time_img_raw_path, "default_time_img_raw_path");
  nh.param<std::string>("output_xy_fit_path", output_xy_fit_path, "default_output_xy_fit_path");
  nh.param<std::string>("output_lmd_pose_path", output_lmd_pose_path, "default_output_lmd_pose_path");
  nh.param<std::string>("color_or_infra", color_or_infra, "default_color_or_infra");
  nh.param<std::string>("output_img_path", output_img_path, "default_output_img_path_path");

  //connect the path
  time_img_predict_depth_grayscale_path = root_path + time_img_predict_depth_grayscale_path;
  time_img_predict_depth_colored_path = root_path + time_img_predict_depth_colored_path;
  time_img_raw_path = root_path + time_img_raw_path;
  lmd_pose_path = root_path + lmd_pose_path;
  output_lmd_pose_path = root_path + output_lmd_pose_path;
  output_xy_fit_path = root_path + output_xy_fit_path;
  output_img_path = root_path + output_img_path;
  printf("time_img_predict_depth_grayscale_path = %s\n", time_img_predict_depth_grayscale_path.c_str());
  printf("read_lmd_poses: %s\n", lmd_pose_path.c_str());
  // read data
  read_lmd_poses(lmd_pose_path);
  std::map<double, std::string> time_img_map = read_images_by_timestamp(time_img_predict_depth_grayscale_path);
  std::map<double, std::string> time_img_color_map = read_images_by_timestamp(time_img_predict_depth_colored_path);
  std::map<double, std::string> time_img_raw_map = read_images_by_timestamp(time_img_raw_path);
  printf("time_img_map.size(): %lu \n", time_img_map.size());
  //coordinate from norm image to FLU world coordinate, the camera image coordinate is right-handed with x pointing right, y pointing down and z pointing forward,
  // the world coordinate is right-handed with x pointing forward, y pointing left and z pointing up
  C_norm2world << 0, 0, 1,
          -1, 0, 0,
          0, -1, 0;
  if(color_or_infra == "color"){
    printf("color camera\n");
    K = K_kun0_camera;
  }else if(color_or_infra == "infra1"){
    printf("infra1 camera\n");
    K = K_kun0_infra1;
  }else if(color_or_infra == "iris0_color") {
    printf("iris0 color camera\n");
    K = K_iris0_color;
  }
  float fx = K.at<float>(0,0);
  float cx = K.at<float>(0,2);
  float fy = K.at<float>(1,1);
  float cy = K.at<float>(1,2);
  printf("fx: %f, fy: %f, cx: %f, cy: %f\n", fx, fy, cx, cy);

  // open output file to write landmark pose data
  std::ofstream file_landmark_pose;
  file_landmark_pose.open(output_lmd_pose_path, std::ios::out);
  file_landmark_pose <<  "# timestamp" << " " << "featID" << " " << "feat_p3d" << " " << "feat_uv_pixel" << " " << "pose_A_t" << " " << "pose_A_q" << std::endl;

  std::map<double, std::vector<Eigen::Vector3d>> points_dense_in_world_map;
  std::map<double, std::vector<Eigen::Vector3d>> points_dense_in_cam_map;
  std::map<double, std::vector<Eigen::Vector3d>> points_sparse_in_world_map;

  for (auto &pose : time_pose_map){
    if(time_img_map.find(pose.first) == time_img_map.end()){
      printf("time: %f, pose not found corresponding img\n", pose.first);
      exit(1);
    }
  }
  printf("pose find all corresponding img, all time_pose_map size = %zu\n", time_pose_map.size());
  sleep(1);
  //check the parameters and data
  queryUserIfContinue();

  int Index = 0; //index for printing
  for(auto &pose : time_pose_map){ //read pose by pose, and read corresponding image by time_img_map, then process the image and pose
    printf("==================[%d] new time = %.3f ==================\n", Index, pose.first);
    depth_image_time = pose.first;
    //read image at this pose time
    cv::Mat depth_image_src = cv::imread(time_img_map[depth_image_time]);
    cv::Mat colored_depth = cv::imread(time_img_color_map[depth_image_time]);
    cv::Mat img_raw = cv::imread(time_img_raw_map[depth_image_time]);
    cv::Mat img_raw_cloned_for_show = img_raw.clone();
    cv::Mat depth_image;
    cv::cvtColor(depth_image_src, depth_image, cv::COLOR_BGR2GRAY);
    // revert depth image, because the depth image is encoded in a way that closer objects have higher pixel values,
    // and farther objects have lower pixel values, which is the opposite of the actual depth.
    // So we need to invert it to get the relative depth of landmarks.
    cv::Mat depth_image_inverted = 255 - depth_image;
    // using mask to set the pixel value of depth_image_inverted to 0 where the pixel value is 255,
    // which means the original depth is 0, which means the sky or very far objects,
    // we want to remove them by setting their depth to 0
    cv::Mat mask;
    cv::compare(depth_image_inverted, 255, mask, cv::CMP_EQ);
    depth_image_inverted.setTo(0, mask);
    // from depth image to acquire the relative depth of landmarks.
    std::vector<Eigen::Vector3d> lmd_in_c_pred_rel_vec;
    std::vector<Eigen::Vector3d> lmd_sparse_known_in_w_vec;
    std::vector<double> lmd_depth_in_c_pred_rel_vec; //predicted co-stereo and self-VIO
    std::vector<double> lmd_depth_in_c_known_vec; //triangulated metric co-stereo and self-VIO
    std::vector<double> lmd_depth_in_c_pred_rel_Collab_vec; //predicted co-stereo
    std::vector<double> lmd_depth_in_c_known_Collab_vec; //triangulated metric co-stereo
    std::vector<double> lmd_depth_in_c_pred_rel_VIO_vec; //predicted self-VIO
    std::vector<double> lmd_depth_in_c_known_VIO_vec; //triangulated metric self-VIO
    std::vector<cv::Point> lmd_uv_vec;

    printf("pose.second.q = %.3f, %.3f, %.3f, %.3f\n", pose.second.q.x(), pose.second.q.y(), pose.second.q.z(), pose.second.q.w());
    printf("pose.second.t = %.3f, %.3f, %.3f\n", pose.second.t[0], pose.second.t[1], pose.second.t[2]);
    cv::Mat depth_image_inverted_color;
    cv::cvtColor(depth_image_inverted, depth_image_inverted_color, cv::COLOR_GRAY2BGR);
    for(auto &lmd : time_lmd_map[depth_image_time]){
      uint8_t known_relative_depth = depth_image_inverted.at<uint8_t>(lmd.second.uv[1], lmd.second.uv[0]);  // Note the depth img pixel is (y, x) format
      if(!known_relative_depth){continue;} //if depth is 0，skip
      Eigen::Vector3d lmd_in_cam_pred_rel = C_norm2world * Eigen::Vector3d(known_relative_depth*(lmd.second.uv[0] - cx)/fx, known_relative_depth*(lmd.second.uv[1] - cy)/fy, known_relative_depth);
      Eigen::Vector3d lmd_sparse_known = pose.second.q.inverse() * (lmd.second.p_in_w - pose.second.t);

      //Here choose OpenVINS VIO points nearby, and collaborative points in the far, because VIO points are more accurate in short range, while collaborative points are more accurate in long range
      if(lmd_sparse_known.x() < 200 && lmd_sparse_known.x() > 1){ //add limitation to the depth of points.
        /** 1. Collaborative + VIO */
        lmd_depth_in_c_known_vec.emplace_back(lmd_sparse_known.x());
        lmd_depth_in_c_pred_rel_vec.emplace_back(lmd_in_cam_pred_rel.x());

        lmd_in_c_pred_rel_vec.emplace_back(lmd_in_cam_pred_rel);
        lmd_sparse_known_in_w_vec.emplace_back(lmd.second.p_in_w);
        lmd_uv_vec.emplace_back(lmd.second.uv[0], lmd.second.uv[1]);

        // plot xy pixel on the image, collaborative landmark using green, vio landmark using red
        if(lmd.first > ID_base){
          lmd_depth_in_c_pred_rel_VIO_vec.emplace_back(lmd_in_cam_pred_rel.x());
          lmd_depth_in_c_known_VIO_vec.emplace_back(lmd_sparse_known.x());
          /** 2. VIO */
//          lmd_depth_in_c_known_vec.emplace_back(lmd_sparse_known.x());
//          lmd_depth_in_c_pred_rel_vec.emplace_back(lmd_in_cam_pred_rel.x());

          cv::circle(depth_image_inverted_color, cv::Point(lmd.second.uv[0], lmd.second.uv[1]), 2, cv::Scalar(0, 0, 255), 2);
          cv::circle(img_raw_cloned_for_show, cv::Point(lmd.second.uv[0], lmd.second.uv[1]), 2, cv::Scalar(0, 0, 255), 2);
          cv::circle(colored_depth, cv::Point(lmd.second.uv[0], lmd.second.uv[1]), 2, cv::Scalar(0, 0, 255), 2);
        }else{
          lmd_depth_in_c_pred_rel_Collab_vec.emplace_back(lmd_in_cam_pred_rel.x());
          lmd_depth_in_c_known_Collab_vec.emplace_back(lmd_sparse_known.x());
          /** 3. Collaborative */
//          lmd_depth_in_c_known_vec.emplace_back(lmd_sparse_known.x());
//          lmd_depth_in_c_pred_rel_vec.emplace_back(lmd_in_cam_pred_rel.x());

          cv::circle(depth_image_inverted_color, cv::Point(lmd.second.uv[0], lmd.second.uv[1]), 2, cv::Scalar(0, 255, 0), 2);
          cv::circle(img_raw_cloned_for_show, cv::Point(lmd.second.uv[0], lmd.second.uv[1]), 2, cv::Scalar(0, 255, 0), 2);
          cv::circle(colored_depth, cv::Point(lmd.second.uv[0], lmd.second.uv[1]), 2, cv::Scalar(0, 255, 0), 2);
        }
      }
    }
    points_sparse_in_world_map[depth_image_time] = lmd_sparse_known_in_w_vec;
    printf("lmd_in_c_pred_rel_vec.size(): %zu | Collab size =  %zu | VIO size = %zu\n",
           lmd_in_c_pred_rel_vec.size(), lmd_depth_in_c_pred_rel_Collab_vec.size(), lmd_depth_in_c_pred_rel_VIO_vec.size());

    //==================== linear fit ===================//
    double map_scale = 1.0;
    double map_shift = 0.0;
    linearFit(lmd_depth_in_c_pred_rel_vec, lmd_depth_in_c_known_vec, map_scale, map_shift);
    std::tuple<double, double> linear_params = {map_scale, map_shift};
    std::cout << "Linear fit parameters:" << std::endl;
    std::cout << "map_scale = " << map_scale << ", map_shift = " << map_shift << std::endl;

    //==================== quad fit ===================//
    auto [quad_a, quad_b, quad_c] = fitQuadraticWithFixedB_C(lmd_depth_in_c_pred_rel_vec, lmd_depth_in_c_known_vec);
    std::cout << "Quadratic fit parameters:" << std::endl;
    std::cout << "quad_a = " << quad_a << ", quad_b = " << quad_b << ", quad_c = " << quad_c << std::endl;
    std::tuple<double, double, double> quad_params = {quad_a, quad_b, quad_c};

    //========================= exponential fit  ===========================//
    auto [sort_x, sort_y, exp_a, exp_b, exp_c, exp_d] = exponential_fit_params(lmd_depth_in_c_pred_rel_vec, lmd_depth_in_c_known_vec);
    std::cout << "Exponential fit parameters:" << std::endl;
    std::cout << "exp_a = " << exp_a << ", exp_b = " << exp_b << ", exp_c = " << exp_d << ", exp_d = " << exp_c << std::endl;
    std::tuple<double, double, double, double> exp_params = {exp_a, exp_b, exp_c, exp_d};
    cv::Mat img_fit = plotXYDataExp(sort_x, sort_y, exp_params);

    //========================= inverse linear fit ===========================//
    std::vector<double> lmd_inverse_depth_in_c_known_vec;
    // convert to inverse depth
    for (auto &val : lmd_depth_in_c_known_vec) {
      if (val != 0) {
        lmd_inverse_depth_in_c_known_vec.emplace_back(1.0 / val);
      } else {
        lmd_inverse_depth_in_c_known_vec.emplace_back(0);
      }
    }
    linearFit(lmd_depth_in_c_pred_rel_vec, lmd_inverse_depth_in_c_known_vec, map_scale, map_shift);
    std::tuple<double, double> inv_linear_params = {map_scale, map_shift};
    std::cout << "Linear fit parameters:" << std::endl;
    std::cout << "map_scale = " << map_scale << ", map_shift = " << map_shift << std::endl;
    cv::Mat img_fit_inv = plotXYDataLinear(lmd_depth_in_c_pred_rel_vec, lmd_inverse_depth_in_c_known_vec, inv_linear_params);

    std::string img_circled_path = output_img_path + "circled_Img_" + std::to_string(depth_image_time) + ".png";
    std::string img_circled_depth_path = output_img_path + "circled_depth_Img_" + std::to_string(depth_image_time) + ".png";
    std::string img_circled_depth_colored_path = output_img_path + "circled_depth_colored_Img_" + std::to_string(depth_image_time) + ".png";
//    cv::imwrite(img_circled_path, img_raw_cloned_for_show);
    cv::imwrite(img_circled_depth_path, depth_image_inverted_color);
    cv::imwrite(img_circled_depth_colored_path, colored_depth);
    cv::imshow("depth_image_with_sparse_lmd", colored_depth);
    cv::moveWindow("depth_image_with_sparse_lmd", 100, 200); // move to (100, 200)
    std::string img_exp_fit_path = output_img_path + "fit_Img_" + std::to_string(depth_image_time) + ".png";
//    cv::imshow("raw_image_with_sparse_lmd", img_raw_cloned_for_show);
    cv::imshow("img_fit", img_fit);
    cv::moveWindow("img_fit", 100, 800);
    cv::imwrite(img_exp_fit_path, img_fit);

    //plot linear, exponential and quadratic fit together
    cv::Mat img_L_E_Q_fit = plotXYDataLinearExpQuad(lmd_depth_in_c_pred_rel_vec, lmd_depth_in_c_known_vec, linear_params, exp_params, quad_params);
    std::string img_L_E_Q_fit_path = output_img_path + "L_E_Q_fit_Img_" + std::to_string(depth_image_time) + ".png";
    cv::imwrite(img_L_E_Q_fit_path, img_L_E_Q_fit);

    // write predicted relative depth and known absolute depth to xy fit file
    std::ofstream ofs(output_xy_fit_path + "Fit_xy.txt", std::ios::out);
    for (int i = 0; i < lmd_depth_in_c_pred_rel_vec.size(); ++i) {
      ofs << std::fixed << std::setprecision(3)
          << pose.first << " "
          << lmd_depth_in_c_pred_rel_vec[i] << " "
          << lmd_depth_in_c_known_vec[i]
          << std::endl;
    }
    ofs.close();

    // write collaborative data to xy fit file
    std::ofstream ofs_collab(output_xy_fit_path + "Fit_xy_Collab.txt", std::ios::out);
    for (int i = 0; i < lmd_depth_in_c_pred_rel_Collab_vec.size(); ++i) {
      ofs_collab << std::fixed << std::setprecision(3)
          << pose.first << " "
          << lmd_depth_in_c_pred_rel_Collab_vec[i] << " "
          << lmd_depth_in_c_known_Collab_vec[i]
          << std::endl;
    }

    // write VIO data to xy fit file
    std::ofstream ofs_vio(output_xy_fit_path + "Fit_xy_VIO.txt", std::ios::out);
    for (int i = 0; i < lmd_depth_in_c_pred_rel_VIO_vec.size(); ++i) {
      ofs_vio << std::fixed << std::setprecision(3)
          << pose.first << " "
          << lmd_depth_in_c_pred_rel_VIO_vec[i] << " "
          << lmd_depth_in_c_known_VIO_vec[i]
          << std::endl;
    }

    // plot the scatter plot of predicted relative depth vs known absolute depth
    std::string filename = output_xy_fit_path + "depth_fit_plot_" + std::to_string(depth_image_time) + ".png";
    plt::scatter(lmd_depth_in_c_pred_rel_vec, lmd_depth_in_c_known_vec, 10); // 10 is the size of points
    plt::title("SLAM landmark scale depths with predicted depths");
    plt::xlabel("Predict Depth (Intensity)");
    plt::ylabel("Scaled Depth (m)");
    plt::grid(true);
    plt::save(filename);
    plt::clf(); // clear figure for the next plot

    // create PCL object and fill in point cloud data
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = 850000;
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.points.resize(cloud.width * cloud.height);
    size_t actual_size = 0; // record the actual number of points added to the cloud
    std::vector<Eigen::Vector3d> points_in_cam;
    std::vector<Eigen::Vector3d> points_in_world;
    // acquire depth image size
    int width = depth_image_inverted.cols;
    int height = depth_image_inverted.rows;
    // set the interval to 2
    int interval = 2;
    // sample points from depth image with the interval
    for (int u = 0; u < width; u += interval) {
      for (int v = 0; v < height; v += interval) {
        // acquire relative depth value from the inverted depth image, which is the output of DepthAnythingV2, and the value is in [0, 255], representing the relative depth from near to far
        uint8_t z = depth_image_inverted.at<uint8_t>(v, u);
        if(!z){
          continue;
        }
        // acquire BGR value for colorizing the point cloud, only for visualization, not used in depth fitting
        cv::Vec3b bgr_pixel = img_raw.at<cv::Vec3b>(v, u);
        uint8_t blue = bgr_pixel[0];
        uint8_t green = bgr_pixel[1];
        uint8_t red = bgr_pixel[2];
        Eigen::Vector3d P_w_in_cam = C_norm2world * Eigen::Vector3d((u - cx)/fx, (v - cy)/fy, 1.0);
        // the function to scale
//        double depth_scaled = map_scale * z + map_shift;//linear fit
//        double depth_scaled = quad_a * (z - quad_b) * (z - quad_b) + quad_c;//quadratic fit
        double depth_scaled = exp_a * std::exp(exp_b * (z - exp_c)) + exp_d;//exponential fit

        Eigen::Vector3d P_w_in_cam_scaled = depth_scaled * P_w_in_cam;
        Eigen::Vector3d P_w_in_world = pose.second.q * P_w_in_cam_scaled + pose.second.t;

        points_in_world.emplace_back(P_w_in_world);
        points_in_cam.emplace_back(P_w_in_cam_scaled);
        // write to PCL point cloud
        pcl::PointXYZRGB pcl_point;
        pcl_point.x = static_cast<float>(P_w_in_cam_scaled.x());
        pcl_point.y = static_cast<float>(P_w_in_cam_scaled.y());
        pcl_point.z = static_cast<float>(P_w_in_cam_scaled.z());
        pcl_point.r = red;
        pcl_point.g = green;
        pcl_point.b = blue;
        cloud.points[actual_size++] = pcl_point;

        //write landmark and pose to file
        file_landmark_pose << fixed << setprecision(3) << pose.first << " " << u*width + v << " " << P_w_in_world.transpose()
        << " " << u << " " << v << " " << pose.second.t.transpose() << " " << pose.second.q.coeffs().transpose() << std::endl;
      }
    }
    file_landmark_pose.close(); // close the file after writing all points

    printf("actual_size = %zu\n", actual_size);
    // store the pointcloud at current timestamp
    points_dense_in_world_map[pose.first] = points_in_world;
    points_dense_in_cam_map[pose.first] = points_in_cam;

    // adjust the size of point cloud to actual_size
    cloud.points.resize(actual_size);
    cloud.width = actual_size; // update width to actual size
    cloud.height = 1;
    // convert PCL point cloud to ROS PointCloud2 message
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(cloud, ros_cloud);
    // set PointCloud msgs
    ros_cloud.header.frame_id = world_frame;
    ros_cloud.header.stamp = ros::Time(depth_image_time);
    // publish XYZ RGB topic
    pointcloud_dense_lmd_in_cam_pub.publish(ros_cloud);

    //publish
    publishPointCloud(points_dense_in_world_map[pose.first], pointcloud_dense_lmd_pub);
    publishPointCloud(points_sparse_in_world_map[pose.first], pointcloud_sparse_lmd_pub);
    publishImages(time_img_raw_map[pose.first], image_raw_pub);

    rate.sleep();
    // this is for debug, wait for user to check the image and point cloud before moving to the next timestamp
    cv::namedWindow("wait for user press key");
    cv::moveWindow("wait for user press key", 100, 1000);
    int key = cv::waitKey(1); // waitKey
    if (key == 'q' || key == 'Q') { // 'q' / 'Q' exit
      cv::destroyAllWindows();
      printf("User press 'q' or 'Q', exit\n");
      exit(-1);
      break;
    }
    sleep(0.2);
    Index++;
  }

  //pub in loop
  while(ros::ok()){
    for(auto &pose : time_pose_map){
      printf("timestamp = %.3f, dense point.size = %zu\n", pose.first, points_dense_in_world_map[pose.first].size());
      publishPointCloud(points_dense_in_world_map[pose.first], pointcloud_dense_lmd_pub);//visualize PointCloud
      publishImages(time_img_color_map[pose.first], image_pub);
      rate.sleep();
    }
    rate.sleep();
  }
    return 0;
}