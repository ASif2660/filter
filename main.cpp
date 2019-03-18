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




#if DEBUG
    std::cout << "Sobel Kernel for gradient along x " << kernel_x;

    std::cout << "Sobel kernel for gradient alogn y " << kernel_y;
#endif

    cv::Mat actual_image = cv::imread(image_name, CV_32F); /* read image */

    simple_edge_detector::edge_detector detect(actual_image);        /* initialize detector */

    detect.show_frame_window();             /*display a short window */

    detect.opencv_to_eigen(global_eigen_matrix);

    std::cout << global_eigen_matrix.cols() << " | " << global_eigen_matrix.rows() << std::endl;

    std::cout << global_eigen_matrix << std::endl;


    // TODO: triangulate with bool on the global_eigen_matrix


    /* Since eigen stores values in column major we use transpose */
    /* double pointers of matrix are not really good in terms of performance on GPU */


    global_eigen_matrix.transposeInPlace(); //converts rows to columns an vice versa

    std::cout << global_eigen_matrix.cols() << " | " << global_eigen_matrix.rows() << std::endl;

    double*  global_vectorized_matrix = global_eigen_matrix.data();

    double*  global_kernel_x = kernel_y.data(); // column major and kernel_y is basically transpose of kernel_x

    double*  global_kernel_y = kernel_x.data();  //row major and kernel_x is basically the transpose

    double*  global_output_matrix = new double[global_eigen_matrix.rows()*global_eigen_matrix.cols()];

    detect.execute_kernels(global_vectorized_matrix, global_kernel_x, global_kernel_y, global_output_matrix);



    /*call the kernel based function */




/*

    double** my_matrix = new double*[global_eigen_matrix.rows()];

    for( int i = 0; i< global_eigen_matrix.rows(); i++){


        my_matrix[i] = new double[global_eigen_matrix.cols()];


    }


    double** my_kernel_x = new double* [kernel_x.rows()];

    for( int i = 0; i< kernel_x.rows(); i++){


        my_kernel_x[i] = new double[kernel_x.cols()];


    }



    double** my_kernel_y = new double* [kernel_y.rows()];

    for( int i = 0; i< kernel_y.rows(); i++){


        my_kernel_y[i] = new double[kernel_y.cols()];


    }


    //double form

    detect.eigen_to_double(my_matrix,global_eigen_matrix);

    detect.eigen_to_double(my_kernel_x, kernel_x);

    detect.eigen_to_double(my_kernel_y, kernel_y);


    for( int i =0; i< kernel_x.rows(); i++)
    {
        for( int j=0; j < kernel_x.cols(); j++){

            //my_matrix[i][j] = kernel_x(i,j);
            std::cout << "Matrix at " << i << " "<< j <<" " << my_kernel_x[i][j] << std::endl;

        }
    }




    double** device_matrix;

    cudaMalloc((void**)&device_matrix, sizeof(double*)*COLS);

    double* temp_pointer[ROWS];


    for( int i =0; i<ROWS; i++ ){

            cudaMalloc((void**)&temp_pointer[i],sizeof(double)* )


    }


*/



return 0;

}




