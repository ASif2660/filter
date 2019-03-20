/*
 * created by Mohammed Asif Chand
 *
 */

#include <iostream>
#include "cuda-support/include/filter.cuh"
#include "eigen/include/eigen3/Eigen/src/Core/Matrix.h"
#include "../../../../usr/local/cuda/include/driver_types.h"
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

#define DEBUG 0




int main(int argc, char* argv[]) {

    const std::string image_name = "x.jpg";

    //MatrixXf m(1080,1307);

    MatrixXf global_eigen_matrix;  /* has dynamic sizing */

    MatrixXf temp_global_matrix;

    MatrixXf kernel_x(3,3);

    MatrixXf kernel_y(3,3);

    kernel_x << -1, 0, 1,
                -2,0,2,
                -1,0,1;

    kernel_y = kernel_x.transpose();

    int* block_size;
    int* min_grid_size;





    cv::Mat actual_image = cv::imread(image_name, CV_32F); /* read image */

    cv::Mat resize_image;

  //  int width_resize = 1020;

  //  int height_resize = 1020;

    cv::resize(actual_image, resize_image, cvSize(COLS, ROWS));

    simple_edge_detector::edge_detector detect(resize_image);        /* initialize detector */

    //detect.show_frame_window();             /*display a short window */

    detect.opencv_to_eigen(global_eigen_matrix);
#if DEBUG
    std::cout << global_eigen_matrix.cols() << " | " << global_eigen_matrix.rows() << std::endl;

    std::cout << global_eigen_matrix << std::endl;
#endif


    // TODO: triangulate with bool on the global_eigen_matrix


    /* Since eigen stores values in column major we use transpose */
    /* double pointers of matrix are not really good in terms of performance on GPU */
    MatrixXf global_temp_matrix = global_eigen_matrix;

    global_eigen_matrix.transposeInPlace(); //converts rows to columns an vice versa
#if DEBUG
    std::cout << global_eigen_matrix.cols() << " | " << global_eigen_matrix.rows() << std::endl;
#endif
    double*  global_vectorized_matrix = global_eigen_matrix.data();
/*
    for(  int i =0; i < ROWS*COLS;  i++)
        std::cout << "The value at " << i << " is : " << global_vectorized_matrix[i] << std::endl;
*/

    double*  global_kernel_x = kernel_y.data(); // column major and kernel_y is basically transpose of kernel_x

    double*  global_kernel_y = kernel_x.data();  //row major and kernel_x is basically the transpose

    auto*  global_output_matrix = new double[ROWS*COLS];

    detect.execute_kernels(global_vectorized_matrix, global_kernel_x, global_kernel_y, global_output_matrix);

  //  std::cout << global_output_matrix[1] << std::endl;

   //  for(  int i =0; i < ROWS*COLS;  i++)
     //   std::cout << "The value at " << i << " is : " << global_output_matrix[i] << std::endl;


    MatrixXf output(ROWS,COLS);

    simple_edge_detector::vector_to_matrix(global_output_matrix, output );


    cv::Mat output_Image(ROWS, COLS, CV_8UC1, CvScalar(0) );

    detect.set_output_eigen_matrix(output);
#if DEBUG
    std::cout << output << std::endl;
#endif
    detect.eigen_to_opencv(output_Image);

    cv::Mat threshold_Image;

    cv::threshold(output_Image,threshold_Image, 50, 175, cv::THRESH_TOZERO_INV);

    cv::namedWindow("The frame window ",CV_WINDOW_AUTOSIZE);

    cv::imshow("Image", threshold_Image);
    // cv::waitKey(0); //display until key is pressed

    cv::waitKey(5000); //display for 5 secs



   // for(int i = 0; i < ROWS; i++)
     //   for(int j =0 ; j < COLS ; j++ ){

    //        std::cout << global_temp_matrix(i,j) << std::endl;
         //   std::cout << global_eigen_matrix(j,i) << std::endl;
     //       std::cout << output(i,j) << std::endl;
    //        std::cout << detect.read_pixel_value(i,j) << std::endl;
    //        std::cout << output_Image.at<double>(i,j) << std::endl;
       //     std::cout << output_Image.cols << output_Image.rows << "Rows and Cols " << std::endl;
      //      std::cout << "Number of channels are " << output_Image.channels() << std::endl;

       // }





return 0;

}




