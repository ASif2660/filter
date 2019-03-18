/*
 * created by Mohammed Asif Chand
 *
 */
#include <iostream>
#include "cuda-support/include/filter.h"
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

static const int COLS = 1307; // this has to be initialized for cuda mem allocation, static intializer errors?
static const int ROWS = 1080;




int main(int argc, char* argv[]) {

    const std::string image_name = "x.jpg";

    //MatrixXf m(1080,1307);

    MatrixXf global_eigen_matrix;  /* has dynamic sizing */

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

    //allocate memory on heap




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

/*
    for( int i =0; i< kernel_x.rows(); i++)
    {
        for( int j=0; j < kernel_x.cols(); j++){

            //my_matrix[i][j] = kernel_x(i,j);
            std::cout << "Matrix at " << i << " "<< j <<" " << my_kernel_x[i][j] << std::endl;

        }
    }

*/


    double** device_matrix;

    cudaMalloc((void**)&device_matrix, sizeof(double*)*COLS);

    double* temp_pointer[ROWS];


    for( int i =0; i<ROWS; i++ ){

            cudaMalloc((void**)&temp_pointer[i],sizeof(double)* )


    }







    /* important declarations for GPU */

    // int totalsize = global_eigen_matrix.cols() * global_eigen_matrix.rows() * sizeof(float);

//     cudaMalloc( (void**)&device_input_matrix, sizeof(m) );

//     cudaMalloc( (void**)&device_output_matrix, sizeof(m) );


// cudaMemcpy(device_input_matrix, global_eigen_matrix, sizeof(m), cudaMemcpyHostToDevice );



  //  cudaCheckErrors("cudaMalloc fail");

     /* note cuda-9.0 has issues with




    /* if( actual_image.empty()){

         exit (1);
     }

     cv::namedWindow("someImage", cv::WINDOW_AUTOSIZE);
     cv::imshow("Displaywindow", actual_image);
     cv::waitKey(0);
     */


    return 0;

}




