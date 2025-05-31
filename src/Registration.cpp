#include "Registration.h"
#include <Eigen/SVD>
#include <iostream>
#include <limits>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
    open3d::io::ReadPointCloud(cloud_source_filename, source_);
    open3d::io::ReadPointCloud(cloud_target_filename, target_);
    source_for_icp_ = source_;
    transformation_ = Eigen::Matrix4d::Identity();
}

Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
    source_ = cloud_source;
    target_ = cloud_target;
    source_for_icp_ = source_;
    transformation_ = Eigen::Matrix4d::Identity();
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

    auto src_pointer = std::make_shared<open3d::geometry::PointCloud>(source_clone);
    auto target_pointer = std::make_shared<open3d::geometry::PointCloud>(target_clone);
    
    open3d::visualization::DrawGeometries({src_pointer, target_pointer});
}

// Execute Iterative Closest Point (ICP) registration algorithm
void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode) {
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
    source_for_icp_ = source_;
    source_for_icp_.Transform(transformation_);
    
    double prev_rmse = std::numeric_limits<double>::max();
    
    // Main ICP iteration loop
    for (int iter = 0; iter < max_iteration; iter++) {
        // Find closest point correspondences between source and target
        auto [src_idx, tgt_idx, current_rmse] = find_closest_point(threshold);
        
        // Check if any valid correspondences were found
        if (src_idx.empty()) {
            std::cout << "ICP: No correspondences found. Stopping at iteration " << iter << std::endl;
            break;
        }
        
        // Check for convergence based on RMSE change
        if (iter > 0 && std::abs(prev_rmse - current_rmse) < relative_rmse * prev_rmse) {
            std::cout << "ICP: Convergence reached at iteration " << iter << std::endl;
            break;
        }
        prev_rmse = current_rmse;
        
        // Compute incremental transformation using SVD
        Eigen::Matrix4d T_inc = get_svd_icp_transformation(src_idx, tgt_idx);
        
        // Apply incremental transformation to working copy
        source_for_icp_.Transform(T_inc);
        
        // Update cumulative transformation matrix
        transformation_ = T_inc * transformation_;
    }
}

// Find closest point correspondences using KD-tree search
std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold) {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Use KDTreeFlann to search the closest target point for each source point.
    // 2. If distance < threshold, record the pair and update RMSE.
    // 3. Return source indices, target indices, and final RMSE.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Build KD-tree for efficient nearest neighbor search on target cloud
    open3d::geometry::KDTreeFlann kd_tree(target_);
    std::vector<size_t> source_indices;
    std::vector<size_t> target_indices;
    double total_sq_dist = 0.0;
    int num_valid = 0;

    // For each point in source cloud, find closest point in target cloud
    for (size_t i = 0; i < source_for_icp_.points_.size(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists_sq(1);
        
        // Search for nearest neighbor
        int num_neighbors = kd_tree.SearchKNN(source_for_icp_.points_[i], 1, indices, dists_sq);
        
        if (num_neighbors > 0) {
            double dist = std::sqrt(dists_sq[0]);
            
            // Only accept correspondences within threshold distance
            if (dist < threshold) {
                source_indices.push_back(i);
                target_indices.push_back(static_cast<size_t>(indices[0]));
                total_sq_dist += dists_sq[0];
                num_valid++;
            }
        }
    }

    // Compute Root Mean Square Error (RMSE)
    double rmse = 0.0;
    if (num_valid > 0) {
        rmse = std::sqrt(total_sq_dist / num_valid);
    }

    return std::make_tuple(source_indices, target_indices, rmse);
}

// Compute optimal transformation using Singular Value Decomposition (SVD)
Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices) {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Compute centroids of source and target points.
    // 2. Subtract centroids and construct matrix H.
    // 3. Use Eigen::JacobiSVD to compute rotation.
    // 4. Handle special reflection case if det(R) < 0.
    // 5. Compute translation t and build 4x4 matrix.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Validate input correspondences
    if (source_indices.size() != target_indices.size() || source_indices.empty()) {
        return Eigen::Matrix4d::Identity();
    }

    size_t n = source_indices.size();
    Eigen::MatrixXd P(n, 3);  // Source points matrix
    Eigen::MatrixXd Q(n, 3);  // Target points matrix

    // Extract corresponding points into matrices
    for (size_t i = 0; i < n; i++) {
        P.row(i) = source_for_icp_.points_[source_indices[i]];
        Q.row(i) = target_.points_[target_indices[i]];
    }

    // Compute centroids of both point sets
    Eigen::Vector3d centroid_P = P.colwise().mean();
    Eigen::Vector3d centroid_Q = Q.colwise().mean();

    // Center the point sets around their centroids
    Eigen::MatrixXd P_centered = P.rowwise() - centroid_P.transpose();
    Eigen::MatrixXd Q_centered = Q.rowwise() - centroid_Q.transpose();

    // Compute cross-covariance matrix H
    Eigen::Matrix3d H = P_centered.transpose() * Q_centered;
    
    // Perform SVD decomposition: H = U * S * V^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Compute rotation matrix R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();

    // Handle reflection case (ensure proper rotation)
    if (R.determinant() < 0) {
        V.col(2) = -V.col(2);  // Flip the last column of V
        R = V * U.transpose();
    }

    // Compute translation vector
    Eigen::Vector3d t = centroid_Q - R * centroid_P;

    // Construct 4x4 homogeneous transformation matrix
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = R;  // Set rotation part
    transformation.block<3, 1>(0, 3) = t;  // Set translation part

    return transformation;
}

// Execute feature-based registration using FPFH descriptors
void Registration::execute_descriptor_registration() {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implement a registration method based entirely on manually implemented feature descriptors.
    // - Preprocess the point clouds (e.g., downsampling).
    // - Detect keypoints in both source and target clouds.
    // - Compute descriptors manually (histogram-based, geometric, etc.) without any built-in functions.
    // - Match descriptors and estimate initial correspondences.
    // - Use RANSAC or other robust method to reject outliers and estimate an initial rigid transformation.
    //   (You may use Open3D's RegistrationRANSACBasedOnCorrespondence() as long as descriptors and matches are computed manually.)
    // - Do NOT use any part of ICP here; this must be a pure descriptor-based initial alignment.
    // - Store the estimated transformation matrix in `transformation_`.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Downsample point clouds for efficiency
    const double voxel_size = 0.05;
    auto source_down = source_.VoxelDownSample(voxel_size);
    auto target_down = target_.VoxelDownSample(voxel_size);

    // Estimate normals for both downsampled clouds
    const double radius_normal = voxel_size * 2.0;
    source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamRadius(radius_normal), 30);
    target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamRadius(radius_normal), 30);

    // Set parameters for FPFH feature computation
    const double radius_feature = voxel_size * 5.0;
    const int bins = 11;  // Standard number of bins for FPFH histograms

    // Lambda function to precompute neighbors for all points (optimization)
    auto precompute_neighbors = [](const open3d::geometry::PointCloud& cloud, 
                                  const open3d::geometry::KDTreeFlann& tree, 
                                  double radius) {
        std::vector<std::pair<std::vector<int>, std::vector<double>>> neighbors(cloud.points_.size());
        // Parallel computation can be enabled here with OpenMP
        for (int i = 0; i < cloud.points_.size(); i++) {
            tree.SearchRadius(cloud.points_[i], radius, neighbors[i].first, neighbors[i].second);
        }
        return neighbors;
    };

    // Build KD-trees and precompute neighbors
    open3d::geometry::KDTreeFlann source_kd_tree(*source_down);
    open3d::geometry::KDTreeFlann target_kd_tree(*target_down);
    
    auto source_neighbors = precompute_neighbors(*source_down, source_kd_tree, radius_feature);
    auto target_neighbors = precompute_neighbors(*target_down, target_kd_tree, radius_feature);

    // Lambda function to compute Simplified Point Feature Histograms (SPFH)
    auto compute_spfh = [bins](const open3d::geometry::PointCloud& cloud,
                                   const std::vector<std::pair<std::vector<int>, std::vector<double>>>& neighbors) {
    std::vector<Eigen::VectorXd> spfh(cloud.points_.size());
    const int descriptor_size = 3 * bins;  // Three angular features, each with 'bins' histogram bins

    // Compute SPFH for each point
    for (int i = 0; i < cloud.points_.size(); i++) {
        const auto& [indices, dists] = neighbors[i];
        Eigen::VectorXd hist = Eigen::VectorXd::Zero(descriptor_size);
        double total_weight = 0.0;
        
        const Eigen::Vector3d& p_s = cloud.points_[i];
        const Eigen::Vector3d& n_s = cloud.normals_[i];

        // Process each neighbor to compute angular features
        for (size_t j = 0; j < indices.size(); j++) {
            const int idx = indices[j];
            if (idx == i) continue;  // Skip self

            const double d_sq = dists[j];
            if (d_sq < 1e-12) continue;  // Skip very close points

            const double d = std::sqrt(d_sq);
            const Eigen::Vector3d& p_t = cloud.points_[idx];
            const Eigen::Vector3d& n_t = cloud.normals_[idx];
            
            // Compute unit vector between points
            Eigen::Vector3d u = (p_t - p_s) / d;
            
            // Compute FPFH angular features
            double alpha = std::acos(std::clamp(n_s.dot(u), -1.0, 1.0));  // Angle between source normal and u
            double phi = std::acos(std::clamp(n_t.dot(u), -1.0, 1.0));    // Angle between target normal and u
            
            // Compute theta (angle around u axis)
            Eigen::Vector3d v = u.cross(n_s).normalized();
            if (v.norm() < 1e-6) continue; // Skip if vectors are parallel
            
            Eigen::Vector3d w = u.cross(v);
            double theta = std::atan2(n_t.dot(v), n_t.dot(w));
            if (theta < 0) theta += 2 * M_PI;  // Ensure positive angle

            // Skip invalid angles
            if (std::isnan(alpha) || std::isnan(phi) || std::isnan(theta)) continue;

            // Weight by inverse distance
            const double weight = 1.0 / d;
            
            // Bin the angular features into histogram
            const int bin_alpha = std::min(static_cast<int>(alpha / M_PI * bins), bins - 1);
            const int bin_phi = std::min(static_cast<int>(phi / M_PI * bins), bins - 1);
            const int bin_theta = std::min(static_cast<int>(theta / (2 * M_PI) * bins), bins - 1);

            // Update histogram bins if indices are valid
            if (bin_alpha >= 0 && bin_alpha < bins &&
                bin_phi >= 0 && bin_phi < bins &&
                bin_theta >= 0 && bin_theta < bins) {
                hist[bin_alpha] += weight;                    // Alpha histogram
                hist[bin_phi + bins] += weight;               // Phi histogram
                hist[bin_theta + 2 * bins] += weight;         // Theta histogram
                total_weight += weight;
            }
        }

        // Normalize histogram by total weight
        if (total_weight > 0) {
            hist /= total_weight;
        }
        spfh[i] = hist;
    }
    return spfh;
};

    // Compute SPFH for both point clouds
    auto source_spfh = compute_spfh(*source_down, source_neighbors);
    auto target_spfh = compute_spfh(*target_down, target_neighbors);

    // Lambda function to compute Fast Point Feature Histograms (FPFH)
    auto compute_fpfh = [bins](const open3d::geometry::PointCloud& cloud,
                              const std::vector<std::pair<std::vector<int>, std::vector<double>>>& neighbors,
                              const std::vector<Eigen::VectorXd>& spfh) {
        std::vector<Eigen::VectorXd> fpfh(cloud.points_.size());
        const int descriptor_size = 3 * bins;

        // Compute FPFH by combining SPFH of point and its neighbors
        for (int i = 0; i < cloud.points_.size(); i++) {
            const auto& [indices, dists] = neighbors[i];
            Eigen::VectorXd fpfh_i = spfh[i];  // Start with SPFH of current point
            const size_t num_neighbors = indices.size() - 1;  // Exclude self

            if (num_neighbors == 0) {
                fpfh[i] = fpfh_i;
                continue;
            }

            // Weight and sum SPFH features of neighbors
            Eigen::VectorXd weighted_sum = Eigen::VectorXd::Zero(descriptor_size);
            double total_weight = 0.0;

            for (size_t j = 0; j < indices.size(); j++) {
                const int idx = indices[j];
                if (idx == i) continue;  // Skip self

                const double d = std::sqrt(dists[j]);
                const double weight = 1.0 / d;
                weighted_sum += weight * spfh[idx];
                total_weight += weight;
            }

            // Combine own SPFH with weighted neighbor SPFH
            fpfh_i += (weighted_sum / num_neighbors);
            fpfh[i] = fpfh_i;
        }
        return fpfh;
    };

    // Compute FPFH features for both point clouds
    auto source_fpfh = compute_fpfh(*source_down, source_neighbors, source_spfh);
    auto target_fpfh = compute_fpfh(*target_down, target_neighbors, target_spfh);

    // Simple keypoint detection (every 10th point for efficiency)
    auto detect_keypoints = [](const open3d::geometry::PointCloud& cloud) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < cloud.points_.size(); i += 10) {
            indices.push_back(i);
        }
        return indices;
    };

    // Extract keypoint indices
    const std::vector<size_t> source_kp_indices = detect_keypoints(*source_down);
    const std::vector<size_t> target_kp_indices = detect_keypoints(*target_down);

    // Extract FPFH descriptors for keypoints only
    std::vector<Eigen::VectorXd> source_kp_descriptors;
    for (const auto idx : source_kp_indices) {
        source_kp_descriptors.push_back(source_fpfh[idx]);
    }

    std::vector<Eigen::VectorXd> target_kp_descriptors;
    for (const auto idx : target_kp_indices) {
        target_kp_descriptors.push_back(target_fpfh[idx]);
    }

    // Build KD-tree for target descriptors for fast matching
    const int descriptor_size = 3 * bins;
    Eigen::MatrixXd target_desc_mat(descriptor_size, target_kp_descriptors.size());
    for (size_t i = 0; i < target_kp_descriptors.size(); i++) {
        target_desc_mat.col(i) = target_kp_descriptors[i];
    }

    // Match descriptors using nearest neighbor search
    open3d::geometry::KDTreeFlann desc_tree(target_desc_mat);
    std::vector<Eigen::Vector2i> correspondences;

    for (size_t i = 0; i < source_kp_descriptors.size(); i++) {
        std::vector<int> indices(2);
        std::vector<double> dists(2);
        desc_tree.SearchKNN(source_kp_descriptors[i], 2, indices, dists);

        // Apply Lowe's ratio test for robust feature matching
        if (dists[0] < 0.8 * dists[1]) {
            correspondences.push_back(Eigen::Vector2i(
                static_cast<int>(source_kp_indices[i]),
                static_cast<int>(target_kp_indices[indices[0]])
            ));
        }
    }

    // RANSAC-based registration using feature correspondences
    const double max_correspondence_dist = voxel_size * 1.5;
    auto estimation = open3d::pipelines::registration::TransformationEstimationPointToPoint(false);
    auto criteria = open3d::pipelines::registration::RANSACConvergenceCriteria(100000, 0.999);

    // Add normal consistency checker for additional robustness
    std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> checkers;
    auto normal_checker = open3d::pipelines::registration::CorrespondenceCheckerBasedOnNormal(0.5236); 
    checkers.push_back(normal_checker);

    // Execute RANSAC registration
    auto result = open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
        *source_down, *target_down, 
        correspondences,
        max_correspondence_dist,
        estimation,
        3,
        checkers,
        criteria
    );

    // Store the computed transformation
    transformation_ = result.transformation_;
}

// Set transformation matrix manually
void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
    transformation_ = init_transformation;
}

// Get current transformation matrix
Eigen::Matrix4d Registration::get_transformation()
{
    return transformation_;
}

// Compute Root Mean Square Error between registered point clouds
double Registration::compute_rmse()
{
    // Build KD-tree for target cloud
    open3d::geometry::KDTreeFlann target_kd_tree(target_);
    
    // Transform source cloud using current transformation
    open3d::geometry::PointCloud source_clone = source_;
    source_clone.Transform(transformation_);
    
    int num_source_points = source_clone.points_.size();
    std::vector<int> idx(1);
    std::vector<double> dist2(1);
    double mse = 0.0;

    // For each transformed source point, find distance to nearest target point
    for (size_t i = 0; i < num_source_points; i++) {
        Eigen::Vector3d source_point = source_clone.points_[i];
        target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
        
        // Update running mean square error
        mse = mse * i / (i + 1) + dist2[0] / (i + 1);
    }
    
    // Return root mean square error
    return sqrt(mse);
}

// Write transformation matrix to file
void Registration::write_tranformation_matrix(std::string filename)
{
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << transformation_;
        outfile.close();
    }
}

// Save merged point cloud after registration
void Registration::save_merged_cloud(std::string filename)
{
    // Create copies and transform source cloud
    open3d::geometry::PointCloud source_clone = source_;
    open3d::geometry::PointCloud target_clone = target_;
    source_clone.Transform(transformation_);
    
    // Merge the two point clouds
    open3d::geometry::PointCloud merged = target_clone + source_clone;
    
    // Save merged cloud to file
    open3d::io::WritePointCloud(filename, merged);
}