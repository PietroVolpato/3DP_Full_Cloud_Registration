#include "Registration.h"
#include <functional>

Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  source_for_icp_ = source_;
}

Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}

void Registration::draw_registration_result()
{
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  Eigen::Vector3d color_s(1, 0.706, 0);
  Eigen::Vector3d color_t(0, 0.651, 0.929);

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
}


void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// For each iteration (for manual ICP implementation):
// 1. Find the closest point correspondences using find_closest_point().
// 2. Use get_svd_icp_transformation() to estimate transformation.
// 3. Apply transformation to source.
// 4. Accumulate transformation and check RMSE convergence.
//
// Note: You are allowed to use Open3D’s RegistrationRANSACBasedOnCorrespondence()
// to compute the initial transformation, if you have already computed descriptors
// and correspondences manually. However, high-level functions like RegistrationICP are not allowed.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Use KDTreeFlann to search the closest target point for each source point.
    // 2. If distance < threshold, record the pair and update RMSE.
    // 3. Return source indices, target indices, and final RMSE.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    open3d::geometry::KDTreeFlann target_kd_tree(target_);
    std::vector<size_t> source_indices;
    std::vector<size_t> target_indices;

    double sum_squared_distances = 0.0;
    size_t valid_correspondeces = 0;

    for (size_t i = 0; i < source_for_icp_.points_.size(); i++) {
      std::vector<int> idx(1);
      std::vector<double> dist(1);

      int n = target_kd_tree.SearchKNN(source_for_icp_.points_[i], 1, idx, dist);
      if (n > 0 && dist[0] < threshold) {
        source_indices.push_back(i);
        target_indices.push_back(static_cast<size_t>(idx[0]));  
        sum_squared_distances += dist[0];
        valid_correspondeces++;

      }
    }

    double rmse = 0.0;
    if (valid_correspondeces > 0) {
      rmse = sqrt(sum_squared_distances / valid_correspondeces);
    }
    
    return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // 1. Compute centroids of source and target points.
  // 2. Subtract centroids and construct matrix H.
  // 3. Use Eigen::JacobiSVD to compute rotation.
  // 4. Handle special reflection case if det(R) < 0.
  // 5. Compute translation t and build 4x4 matrix.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // 1. Compute centroids
  Eigen::Vector3d source_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_centroid = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < source_indices.size(); i++) {
    source_centroid += source_for_icp_.points_[source_indices[i]];
    target_centroid += target_.points_[target_indices[i]];
  }

  source_centroid /= source_indices.size();
  target_centroid /= target_indices.size();

  // 2. Subtract centroids and construct matrix H
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);

  for (size_t i = 0; i < source_indices.size(); i++) {
    Eigen::Vector3d source_point = source_for_icp_.points_[source_indices[i]] - source_centroid;
    Eigen::Vector3d target_point = target_.points_[target_indices[i]] - target_centroid;
    H += source_point * target_point.transpose();
  }

  // 3. Use Eigen::JacobiSVD to compute rotation
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();

  Eigen::Matrix3d R = V * U.transpose();

  // 4. Handle special reflection case
  if (R.determinant() < 0) {
    V.col(2) *= -1;
    R = V * U.transpose();
  }

  // 5. Compute translation t
  Eigen::Vector3d t = target_centroid - R * source_centroid;

  // Build the transformation matrix
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4, 4);
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;

  return transformation;
}

// Helper function to calculate Darboux frame features for a point pair
std::tuple<double, double, double> computeAngularFeatures (const Eigen::Vector3d& s, const Eigen::Vector3d& n_s,
                                                           const Eigen::Vector3d& t, const Eigen::Vector3d& n_t) {
  Eigen::Vector3d u = n_s;
  Eigen::Vector3d d = (t - s).normalized();
  Eigen::Vector3d v = d.cross(u).normalized();
  Eigen::Vector3d w = u.cross(v).normalized();

  double alpha = v.dot(n_t);
  double phi = u.dot(d);
  double theta = std::atan2(w.dot(n_t), u.dot(n_t));

  return {alpha, phi, theta};
}

double compute_l2_distance(const std::vector<double>& desc1, const std::vector<double>& desc2) {
    double sum = 0.0;
    for (size_t i = 0; i < desc1.size(); ++i) {
        double diff = desc1[i] - desc2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void Registration::execute_descriptor_registration()
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implement a registration method based entirely on manually implemented feature descriptors.
// - Preprocess the point clouds (e.g., downsampling).
// - Detect keypoints in both source and target clouds.
// - Compute descriptors manually (histogram-based, geometric, etc.) without any built-in functions.
// - Match descriptors and estimate initial correspondences.
// - Use RANSAC or other robust method to reject outliers and estimate an initial rigid transformation.
//   (You may use Open3D’s RegistrationRANSACBasedOnCorrespondence() as long as descriptors and matches are computed manually.)
// - Do NOT use any part of ICP here; this must be a pure descriptor-based initial alignment.
// - Store the estimated transformation matrix in `transformation_`.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // 1. Perform a downsampling of the point clouds.
  const double voxel_size = 0.05;
  std::shared_ptr<open3d::geometry::PointCloud> source_downsampled = source_.VoxelDownSample(voxel_size);
  std::shared_ptr<open3d::geometry::PointCloud> target_downsampled = target_.VoxelDownSample(voxel_size);

  // 2. Estimate normals
  source_downsampled->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));
  target_downsampled->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));

  // 3. Compute SFPFH histograms
  const double radius = 0.1;
  const int bins = 11;
  const double bin_size = 2.0 / bins;
  const double theta_bin = 2 * M_PI / bins;
  open3d::geometry::KDTreeFlann source_kd_tree(*source_downsampled);
  open3d::geometry::KDTreeFlann target_kd_tree(*target_downsampled);
  
  auto compute_sfpfh = [&](std::shared_ptr<open3d::geometry::PointCloud> cloud,
                           open3d::geometry::KDTreeFlann& kd_tree,
                           std::vector<std::vector<double>>& sfpfh) {
    size_t num_points = cloud->points_.size();
    sfpfh.resize(num_points, std::vector<double>(bins * 3, 0.0));
    std::vector<int> idx;
    std::vector<double> dist;
    for (size_t i = 0; i < num_points; i++) {
      Eigen::Vector3d p = cloud->points_[i];
      Eigen::Vector3d n_p = cloud->normals_[i];
      int n = kd_tree.SearchRadius(p, radius, idx, dist);
      for (size_t j = 0; j < n; j++) {
        size_t k = idx[j];
        auto [alpha, phi, theta] = computeAngularFeatures(p, n_p, cloud->points_[k], cloud->normals_[k]);
        int alpha_bin = std::min(bins - 1, int((alpha + 1) / bin_size));
        int phi_bin = std::min(bins - 1, int((phi + 1) / bin_size));
        int theta_bin = std::min(bins - 1, int((theta + M_PI) / bin_size));
        sfpfh[i][alpha_bin] += 1.0;
        sfpfh[i][bins + phi_bin] += 1.0;
        sfpfh[i][2 * bins + theta_bin] += 1.0;
      }
      // Normalize the histogram
      double sum = 0.0;
      for (double val : sfpfh[i]) sum += val;
      if (sum > 0) for (double& val : sfpfh[i]) val /= sum;
    }
  };

  std::vector<std::vector<double>> source_sfpfh, target_sfpfh;
  compute_sfpfh(source_downsampled, source_kd_tree, source_sfpfh);
  compute_sfpfh(target_downsampled, target_kd_tree, target_sfpfh);

  // 4. Compute the FPFH 
  auto compute_fpfh = [&](std::shared_ptr<open3d::geometry::PointCloud> cloud,
                          open3d::geometry::KDTreeFlann& kd_tree,
                          const std::vector<std::vector<double>>& sfpfh,
                          std::vector<std::vector<double>>& fpfh) {
    size_t num_points = cloud->points_.size();
    fpfh = sfpfh;
    std::vector<int> idx;
    std::vector<double> dist;
    for (size_t i = 0; i < num_points; i++) {
      Eigen::Vector3d p = cloud->points_[i];
      int n = kd_tree.SearchRadius(p, radius, idx, dist);
      double weight_sum = 0.0;
      std::vector<double> acc(bins * 3, 0.0);
      for (size_t j = 0; j < n; j++) {
        double weight = 1.0 / (dist[j] + 1e-6);
        weight_sum += weight;
        auto& neighbor = sfpfh[idx[j]];
        for (size_t k = 0; k < bins * 3; k++) acc[k] += weight * neighbor[k];
      }
      if (weight_sum > 0) {
        for (size_t k = 0; k < bins * 3; k++) fpfh[i][k] = acc[k] / weight_sum;
      }
      // Normalize the histogram
      double sum = 0.0;
      for (double val : fpfh[i]) sum += val;
      if (sum > 0) for (double& val : fpfh[i]) val /= sum;
    }
  };

  std::vector<std::vector<double>> source_fpfh, target_fpfh;
  compute_fpfh(source_downsampled, source_kd_tree, source_sfpfh, source_fpfh);
  compute_fpfh(target_downsampled, target_kd_tree, target_sfpfh, target_fpfh);

  // 5. Match descriptors using manual distance computation
  std::vector<Eigen::Vector2i> correspondences;
  
  // For each source descriptor, find the best match in target descriptors
  for (size_t i = 0; i < source_fpfh.size(); ++i) {
    double best_distance = std::numeric_limits<double>::max();
    int best_match = -1;
    
    for (size_t j = 0; j < target_fpfh.size(); ++j) {
      double distance = compute_l2_distance(source_fpfh[i], target_fpfh[j]);
      if (distance < best_distance) {
        best_distance = distance;
        best_match = j;
      }
    }
    
    if (best_match >= 0) {
      correspondences.emplace_back(i, best_match);
    }
  }

  // 6. Estimate initial transformation using RANSAC
  open3d::pipelines::registration::RegistrationResult result = open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
    *source_downsampled, *target_downsampled, correspondences,
    voxel_size * 1.5,
    open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
    3,
    std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>>{},
    open3d::pipelines::registration::RANSACConvergenceCriteria(4000000, 500));
  transformation_ = result.transformation_;
}



void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_ = init_transformation;
}

Eigen::Matrix4d Registration::get_transformation()
{
  return transformation_;
}


double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points = source_clone.points_.size();
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse = 0.0;

  for (size_t i = 0; i < num_source_points; ++i) {
    Eigen::Vector3d source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i / (i + 1) + dist2[0] / (i + 1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile(filename);
  if (outfile.is_open()) {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;
  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone + source_clone;
  open3d::io::WritePointCloud(filename, merged);
}
