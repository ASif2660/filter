//
// Created by asif on 07.03.19.
// simple example applies convolution with cuda cores using both eigen and opencv interface
//

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

  void vector_to_matrix(double* vector, double** matrix, int size_of_vector, int rows, int cols);






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


        void eigen_to_opencv(cv::Mat& cv_image );

        void  opencv_to_eigen(MatrixXf& eigen_matrix);

        int get_frame_width_cv();

        int get_frame_height_cv();

        void eigen_to_double(double** matrix, MatrixXf eigen_matrix);


    private:

         cv::Mat* _Image;

         MatrixXf _eigenMatrix; //global matrix from image
         MatrixXf _dMatrix; //device matrix

         int _wrows;
         int _hcols; //this reflects the size of the image


      //   sobel_operator* _op;
      //   sobel_operator* _dop; //device memory



    };







} //namespace


#endif //FILTER_FILTER_H
