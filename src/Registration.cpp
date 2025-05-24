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
  // Initialize variables for ICP iteration
  double prev_rmse = std::numeric_limits<double>::max();
  transformation_ = Eigen::Matrix4d::Identity();  // Start with identity transformation
  source_for_icp_ = source_;  // Reset source cloud for ICP processing
  
  // Main ICP iteration loop
  for (size_t i = 0; i < max_iteration; i++) {
    // 1. Find closest point correspondences between source and target
    auto [source_indices, target_indices, rmse] = find_closest_point(threshold);
    
    // 2. Check for convergence based on RMSE change
    if (std::abs(prev_rmse - rmse) < relative_rmse) {
      std::cout << "Converged after " << i << " iterations with RMSE: " << rmse << std::endl;
      break;
    }
    prev_rmse = rmse;  // Update previous RMSE for next iteration comparison
    
    // 3. Estimate transformation using SVD-based point-to-point ICP
    Eigen::Matrix4d transformation = get_svd_icp_transformation(source_indices, target_indices);
    
    // 4. Apply transformation to source cloud for next iteration
    source_for_icp_.Transform(transformation);
    
    // 5. Accumulate transformations (T_total = T_current * T_previous)
    transformation_ = transformation * transformation_;
  }
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Use KDTreeFlann to search the closest target point for each source point.
    // 2. If distance < threshold, record the pair and update RMSE.
    // 3. Return source indices, target indices, and final RMSE.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Build KD-Tree for efficient nearest neighbor search in target point cloud
    open3d::geometry::KDTreeFlann target_kd_tree(target_);
    
    // Initialize containers for storing valid correspondence pairs
    std::vector<size_t> source_indices;
    std::vector<size_t> target_indices;
    
    // Initialize variables for RMSE calculation
    double mse = 0.0;
    size_t count = 0;
    std::vector<int> idx(1);      // Storage for nearest neighbor index
    std::vector<double> dist(1);  // Storage for squared distance to nearest neighbor

    // Iterate through all points in the source cloud
    for (size_t i = 0; i < source_for_icp_.points_.size(); i++) {
      // Find the closest point in target cloud using KD-Tree
      target_kd_tree.SearchKNN(source_for_icp_.points_[i], 1, idx, dist);
      
      // Check if the distance is within the threshold (note: dist[0] is squared distance)
      if (dist[0] < threshold * threshold) {
      // Store the correspondence pair
      source_indices.push_back(i);
      target_indices.push_back(idx[0]);
      
      // Update running mean squared error using incremental formula
      mse = ((count * mse) + dist[0]) / (count + 1);
      count++;
      }
    }
    
    // Calculate RMSE from MSE, or return max value if no valid correspondences found
    double rmse = count > 0 ? std::sqrt(mse) : std::numeric_limits<double>::max();
    
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
  size_t N = source_indices.size();
  Eigen::MatrixXd S(3, N), T(3, N);
  for (size_t i = 0; i < N; i++) {
    S.col(i) = source_for_icp_.points_[source_indices[i]];
    T.col(i) = target_.points_[target_indices[i]];
  }
  Eigen::Vector3d centroid_s = S.rowwise().mean();
  Eigen::Vector3d centroid_t = T.rowwise().mean();

  // 2. Center the points
  Eigen::MatrixXd S_centered = S.colwise() - centroid_s;
  Eigen::MatrixXd T_centered = T.colwise() - centroid_t;

  Eigen::Matrix3d H = S_centered * T_centered.transpose();

  // 3. Compute SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  // Get the rotation matrix
  Eigen::Matrix3d R = V * U.transpose();

  // 4. Handle reflection case
  if (R.determinant() < 0) {
    V.col(2) *= -1; // Flip the last column of V
    R = V * U.transpose();
  }

  // 5. Compute translation
  Eigen::Vector3d t = centroid_t - R * centroid_s;

  // Build the transformation matrix
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;

  return transformation;
}

// Helper function to calculate Darboux frame features for a point pair
// Helper function to calculate Darboux frame features for a point pair
// This computes the angular features (alpha, phi, theta) used in FPFH descriptors
// based on the Darboux frame constructed from two points and their normals
std::tuple<double, double, double> computeAngularFeatures (const Eigen::Vector3d& s, const Eigen::Vector3d& n_s,
                              const Eigen::Vector3d& t, const Eigen::Vector3d& n_t) {
  // Build Darboux frame at point s
  Eigen::Vector3d u = n_s.normalized();  // Normal at source point
  Eigen::Vector3d d = (t - s).normalized();  // Direction vector from source to target
  Eigen::Vector3d v = d.cross(u).normalized();  // First tangent vector (perpendicular to u and d)
  Eigen::Vector3d w = u.cross(v);  // Second tangent vector (completes the orthogonal frame)

  // Compute the three angular features
  double alpha = v.dot(n_t);  // Angle between v and target normal (measures surface variation)
  double phi = u.dot(d);      // Angle between source normal and connection vector (measures surface orientation)
  double theta = std::atan2(w.dot(n_t), u.dot(n_t));  // Rotation angle around the connection vector

  return {alpha, phi, theta};
}

// Helper function to compute L2 (Euclidean) distance between two feature descriptors
// Used for matching descriptors between source and target point clouds
double compute_l2_distance(const std::vector<double>& desc1, const std::vector<double>& desc2) {
  double sum = 0.0;
  // Sum squared differences between corresponding elements
  for (size_t i = 0; i < desc1.size(); ++i) {
    double diff = desc1[i] - desc2[i];
    sum += diff * diff;
  }
  // Return square root of sum (Euclidean distance)
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
  // 1. Perform a downsampling of the point clouds to reduce computational complexity
  const double voxel_size = 0.05;
  std::shared_ptr<open3d::geometry::PointCloud> source_downsampled = source_.VoxelDownSample(voxel_size);
  std::shared_ptr<open3d::geometry::PointCloud> target_downsampled = target_.VoxelDownSample(voxel_size);

  // 2. Estimate normals for both point clouds using KD-tree hybrid search
  // Normals are required for computing geometric features
  source_downsampled->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));
  target_downsampled->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));

  // 3. Setup parameters for FPFH (Fast Point Feature Histogram) computation
  const double radius = 0.1; // Radius for neighborhood search
  const int bins = 11; // Number of bins for each feature (alpha, phi, theta)
  const double bin_size = 2.0 / bins; // Size of each bin for normalization (alpha and phi range [-1,1])
  const double theta_bin = 2 * M_PI / bins;  // Size of theta bin for normalization (theta range [0, 2π])
  
  // Build KD-trees for efficient nearest neighbor search
  open3d::geometry::KDTreeFlann source_kd_tree(*source_downsampled);
  open3d::geometry::KDTreeFlann target_kd_tree(*target_downsampled);
  
  // Lambda function to compute Simplified Fast Point Feature Histogram (SFPFH) features
  // SFPFH is the first step of FPFH computation - computes local features for each point
  auto compute_sfpfh = [&](std::shared_ptr<open3d::geometry::PointCloud> cloud,
                           open3d::geometry::KDTreeFlann& kd_tree,
                           std::vector<std::vector<double>>& sfpfh) {
    size_t num_points = cloud->points_.size();
    sfpfh.resize(num_points, std::vector<double>(bins * 3, 0.0)); // 3 features × bins each
    std::vector<int> idx;
    std::vector<double> dist;
    
    // For each point in the cloud
    for (size_t i = 0; i < num_points; i++) {
      Eigen::Vector3d p = cloud->points_[i];
      Eigen::Vector3d n_p = cloud->normals_[i];
      
      // Find all neighbors within the specified radius
      int n = kd_tree.SearchRadius(p, radius, idx, dist);
      if (idx.size() < 5) continue; // Skip points with too few neighbors
      
      // For each neighbor, compute angular features and update histogram
      for (size_t j = 1; j < idx.size(); ++j) { // Skip self (idx[0])
        size_t k = idx[j];
        // Compute Darboux frame features between current point and neighbor
        auto [alpha, phi, theta] = computeAngularFeatures(p, n_p, cloud->points_[k], cloud->normals_[k]);
        
        // Quantize features into histogram bins
        int alpha_bin = std::min(bins - 1, int((alpha + 1) / bin_size));  // alpha ∈ [-1,1]
        int phi_bin = std::min(bins - 1, int((phi + 1) / bin_size));      // phi ∈ [-1,1]
        int theta_bin = std::min(bins - 1, int((theta + M_PI) / bin_size)); // theta ∈ [0,2π]
        
        // Increment corresponding histogram bins
        sfpfh[i][alpha_bin] += 1.0;
        sfpfh[i][bins + phi_bin] += 1.0;
        sfpfh[i][2 * bins + theta_bin] += 1.0;
      }
      
      // Normalize the histogram to make it scale-invariant
      double sum = 0.0;
      for (double val : sfpfh[i]) sum += val;
      if (sum > 0) for (double& val : sfpfh[i]) val /= sum;
    }
  };

  // Compute SFPFH for both source and target point clouds
  std::vector<std::vector<double>> source_sfpfh, target_sfpfh;
  compute_sfpfh(source_downsampled, source_kd_tree, source_sfpfh);
  compute_sfpfh(target_downsampled, target_kd_tree, target_sfpfh);

  // 4. Compute the final FPFH (Fast Point Feature Histogram) descriptors
  // FPFH enhances SFPFH by incorporating weighted contributions from neighboring SFPFH
  auto compute_fpfh = [&](std::shared_ptr<open3d::geometry::PointCloud> cloud,
                          open3d::geometry::KDTreeFlann& kd_tree,
                          const std::vector<std::vector<double>>& sfpfh,
                          std::vector<std::vector<double>>& fpfh) {
    size_t num_points = cloud->points_.size();
    fpfh = sfpfh; // Initialize FPFH with SFPFH values
    std::vector<int> idx;
    std::vector<double> dist;
    
    // For each point, enhance its descriptor with weighted neighbor contributions
    for (size_t i = 0; i < num_points; i++) {
      Eigen::Vector3d p = cloud->points_[i];
      int n = kd_tree.SearchRadius(p, radius, idx, dist);
      
      double weight_sum = 0.0;
      std::vector<double> acc(bins * 3, 0.0); // Accumulator for weighted neighbor contributions
      
      // Accumulate weighted SFPFH from all neighbors
      for (size_t j = 0; j < n; j++) {
        double weight = 1.0 / (dist[j] + 1e-6); // Inverse distance weighting
        weight_sum += weight;
        auto& neighbor = sfpfh[idx[j]];
        for (size_t k = 0; k < bins * 3; k++) acc[k] += weight * neighbor[k];
      }
      
      // Normalize by total weight and store as final FPFH
      if (weight_sum > 0) {
        for (size_t k = 0; k < bins * 3; k++) fpfh[i][k] = acc[k] / weight_sum;
      }
      
      // Final normalization to ensure descriptor is unit-normalized
      double sum = 0.0;
      for (double val : fpfh[i]) sum += val;
      if (sum > 0) for (double& val : fpfh[i]) val /= sum;
    }
  };

  // Compute final FPFH descriptors for both point clouds
  std::vector<std::vector<double>> source_fpfh, target_fpfh;
  compute_fpfh(source_downsampled, source_kd_tree, source_sfpfh, source_fpfh);
  compute_fpfh(target_downsampled, target_kd_tree, target_sfpfh, target_fpfh);

  /*
  // Alternative: Manual descriptor matching (slower but more explicit)
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
  }*/
  
  // 5. Match descriptors efficiently using KD-tree in descriptor space
  // Convert target descriptors to matrix format for KD-tree construction
  Eigen::MatrixXd matched_target(target_fpfh[0].size(), target_fpfh.size());
  for (size_t i = 0; i < target_fpfh.size(); i++)
    matched_target.col(i) = Eigen::Map<const Eigen::VectorXd>(target_fpfh[i].data(), target_fpfh[i].size());

  // Build KD-tree in descriptor space for fast nearest neighbor search
  open3d::geometry::KDTreeFlann descriptor_kd_tree(matched_target);
  std::vector<Eigen::Vector2i> correspondences;
  
  // For each source descriptor, find its nearest neighbor in target descriptors
  for (size_t i = 0; i < source_fpfh.size(); i++) {
    std::vector<int> idx(1);
    std::vector<double> dist(1);
    Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(source_fpfh[i].data(), source_fpfh[i].size());
    descriptor_kd_tree.SearchKNN(q, 1, idx, dist);
    if (!idx.empty()) correspondences.emplace_back(i, idx[0]);
  }

  // 6. Estimate initial transformation using RANSAC-based correspondence matching
  // RANSAC removes outlier correspondences and estimates robust transformation
  open3d::pipelines::registration::RegistrationResult result = open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
    *source_downsampled, *target_downsampled, correspondences,
    voxel_size * 1.5,  // Maximum correspondence distance threshold
    open3d::pipelines::registration::TransformationEstimationPointToPoint(false), // Point-to-point estimation
    3,  // Minimum number of correspondences for valid transformation
    std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>>{}, // No additional checkers
    open3d::pipelines::registration::RANSACConvergenceCriteria(4000000, 500)); // Max iterations and confidence
  
  // Store the estimated transformation matrix
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