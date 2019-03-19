//
// Created by asif on 07.03.19.
// simple example applies convolution with cuda cores using both eigen and opencv interface
//

// do not use cuda specific functions in header it won't work.
#ifndef _FILTER_H_
#define _FILTER_H_

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include "../../../../../../usr/local/cuda/include/driver_types.h"

#define COLS 1307 // this has to be initialized for cuda mem allocation, static intializer errors?
#define ROWS 1080
#define GRIDSIZE 200
#define BLOCKSIZE 500

//define a global method here and use it inside the class below

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXf; //simplifies the naming






namespace simple_edge_detector{

  /*  struct sobel_operator{

        Matrix33f s_x << -1,0,1,
                        -2,0,2,
                        -1,0,1;


        Matrix33f s_y << -1,-2,-1,
                          0,0,0,
                          1,2,1;

//hard coded fields
    };
*/






  void print_matrix_pointer(double** matrix, int rows, int cols);

  void print_block_matrix (MatrixXf some_eigen_matrix, int block_rows, int block_cols);

  void vector_to_matrix(double* vector, MatrixXf& eigen_output);


  __global__
  void edge_detector_gpu (double* vectorized_matrix, double* kernel_x, double* kernel_y, double* output_vector_matrix);






    class edge_detector {

    public:

        edge_detector(cv::Mat &frame) {

            _Image = new cv::Mat; // initialise the pointer, otherwise it's gonna be dangling.

            cv::cvtColor(frame, *(_Image), CV_BGR2GRAY);

            _wrows = _Image->rows;

            _hcols = _Image->cols;

          //   _op = new sobel_operator;

        }


        ~edge_detector(){

            delete _Image;

        //    delete _op;

        }


        void show_frame_window(){

            cv::namedWindow("The frame window ",CV_WINDOW_AUTOSIZE);
            cv::imshow("Image", *(_Image)  );
           // cv::waitKey(0); //display until key is pressed
           cv::waitKey(5000); //display for 5 secs

        }



        double read_pixel_value (int i, int j){

         return _Image->at<char>(i,j);

        }

        void eigen_to_opencv(cv::Mat& output );

        void  opencv_to_eigen(MatrixXf& eigen_matrix);

        void set_output_eigen_matrix(MatrixXf& eigen_matrix);

        int get_frame_width_cv();

        int get_frame_height_cv();

        void eigen_to_double(double** matrix, MatrixXf eigen_matrix);


        void execute_kernels(double* global_vectorized_matrix, double* global_kernel_x, double* global_kernel_y,

                double* global_output_matrix);

    private:

         cv::Mat* _Image;

         MatrixXf _eigenMatrix; //global matrix from image

         int _wrows;

         int _hcols; //this reflects the size of the image

         double*  device_gpu_output;

         double* device_vectorized_matrix;

         double* device_kernel_x;

         double* device_kernel_y;




        //   sobel_operator* _op;
      //   sobel_operator* _dop; //device memory



    };







} //namespace


#endif //FILTER_FILTER_H
