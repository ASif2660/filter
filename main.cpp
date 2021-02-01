/*
 * created by Mohammed Asif Chand
 *
 */

#include <iostream>
#include "mylibrary/include/filter.cuh"
#include "eigen/include/eigen3/Eigen/src/Core/Matrix.h"
#include "cuda/include/driver_types.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Core>


#pragma hd_warning_disable
// this program is for testing





int main(int argc, char* argv[]) {

    
    double*  global_vectorized_matrix = global_eigen_matrix.data();

    double*  global_kernel_x = kernel_y.data(); // column major and kernel_y is basically transpose of kernel_x

    double*  global_kernel_y = kernel_x.data();  //row major and kernel_x is basically the transpose

    auto*  global_output_matrix = new double[ROWS*COLS];

    const std::string IMAGE_NAME = "x.jpg";
 
    int* block_size;

    int* min_grid_size;

    MatrixXf output(ROWS,COLS);

    MatrixXf global_eigen_matrix;  /* has dynamic sizing */

    MatrixXf temp_global_matrix;

    MatrixXf kernel_x(3,3);

    MatrixXf kernel_y(3,3);

    kernel_x << -1, 0, 1,
                -2,0,2,
                -1,0,1;

    kernel_y = kernel_x.transpose();


    cv::Mat actual_image = cv::imread(IMAGE_NAME, CV_32F); /* read image */

    cv::Mat resize_image;

    cv::resize(actual_image, resize_image, cvSize(COLS, ROWS));

    simple_edge_detector::edge_detector detect(resize_image);        /* initialize detector */

    //detect.show_frame_window();             /*display a short window */

     detect.opencv_to_eigen(global_eigen_matrix);

    // TODO: triangulate with bool on the global_eigen_matrix
    /* Since eigen stores values in column major we use transpose */
    /* double pointers of matrix are not really good in terms of performance on GPU */
   
    MatrixXf global_temp_matrix = global_eigen_matrix;

    global_eigen_matrix.transposeInPlace(); //converts rows to columns an vice versa    

    detect.execute_kernels(global_vectorized_matrix, global_kernel_x, global_kernel_y, global_output_matrix);

    simple_edge_detector::vector_to_matrix(global_output_matrix, output );

    cv::Mat output_Image(ROWS, COLS, CV_8UC1, CvScalar(0) );

    detect.set_output_eigen_matrix(output);
    
    detect.eigen_to_opencv(output_Image);

    cv::Mat threshold_Image;

    cv::threshold(output_Image,threshold_Image, 50, 175, cv::THRESH_TOZERO_INV);

    cv::namedWindow("The frame window ",CV_WINDOW_AUTOSIZE);

    cv::imshow("Image", threshold_Image);
    
    cv::waitKey(5000); //display for 5 secs



return 0;

}




