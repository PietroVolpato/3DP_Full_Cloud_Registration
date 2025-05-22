#include <iostream>
#include "Registration.h"


int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <source.ply> <target.ply>" << std::endl;
        return -1;
    }

    Registration registration(argv[1], argv[2]);

    auto start_desc = std::chrono::high_resolution_clock::now();
    registration.execute_descriptor_registration();
    auto end_desc = std::chrono::high_resolution_clock::now();

    std::cout << "Initial RMSE (after descriptor-based alignment): " << registration.compute_rmse() << std::endl;

    auto start_icp = std::chrono::high_resolution_clock::now();
    

    registration.execute_icp_registration(0.2, 100, 1e-6, "svd");
    auto end_icp = std::chrono::high_resolution_clock::now();

    registration.draw_registration_result();

    double time_desc = std::chrono::duration<double, std::milli>(end_desc - start_desc).count();
    double time_icp = std::chrono::duration<double, std::milli>(end_icp - start_icp).count();

    std::cout << "Descriptor-based Registration Time: " << time_desc << " ms" << std::endl;
    std::cout << "ICP Refinement Time: " << time_icp << " ms" << std::endl;
    std::cout << "Final RMSE: " << registration.compute_rmse() << std::endl;

    registration.write_tranformation_matrix("transformation_final.txt");
    registration.save_merged_cloud("merged_final.ply");

    return 0;
}

