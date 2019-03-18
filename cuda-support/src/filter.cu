//
// Created by asif on 07.03.19.
//

#include "../include/filter.h"







void simple_edge_detector::edge_detector::eigen_to_opencv(cv::Mat& cv_image ) {


    // uses the grey scale or one channel matrix for conversion, careful do not use color image

    //precondition  is your input eigen matrix as the _Image is already resolved in constructor

    cv::eigen2cv(_eigenMatrix, cv_image);


}



void simple_edge_detector::edge_detector::opencv_to_eigen(MatrixXf& eigen_matrix) {

    //precondition is your input image which is gonna be greyscale, as _eigenMatrix is built from eigen_to_opencv

    cv::cv2eigen(*_Image, eigen_matrix);


}


void simple_edge_detector::edge_detector::eigen_to_double(double **matrix, MatrixXf eigen_matrix) {

/*
     matrix = new double*[eigen_matrix.rows()];

    for( int i = 0; i< eigen_matrix.rows(); i++){


        matrix[i] = new double[eigen_matrix.cols()];

    }
*/
    //copy data

    for( int j =0; j< eigen_matrix.cols(); j++)
    {
        for( int i=0; i < eigen_matrix.rows(); i++){

            matrix[i][j] = eigen_matrix(i,j);

           // std::cout << " The value at " << matrix[i][j] << std:endl; // this line has an issue with cuda I guess but it works

        }
    }




}





//getters and setters for

int simple_edge_detector::edge_detector::get_frame_height_cv() {

    return _hcols;
}

int simple_edge_detector::edge_detector::get_frame_width_cv() {


    return _wrows;

}


void simple_edge_detector::print_matrix_pointer( double** matrix, int rows, int cols ){


    for( int j =0; j <  cols; j++ )
    {
        for( int i=0; i< rows; i++ ){


            std::cout << "The value at index " << i << " and " << j << "is" << matrix[i][j] << std::endl;

        }
    }

}

void simple_edge_detector::print_block_matrix(MatrixXf some_eigen_matrix, int block_rows, int block_cols){


    std::cout << "Block of size " << block_rows << "and "<< block_cols << "is" << std::endl;

     some_eigen_matrix.block(0,0, block_rows, block_cols); // this is throwing some error



}



void simple_edge_detector::vector_to_matrix(double* vector, double** matrix, int size_of_vector, int rows, int cols){


    // this basically converts the vector to matrix of required size

    if( size_of_vector !=  rows * cols ){


        std::cout << " This is an illegal conversion !!!, Please check the total size, rows and cols " << std::endl;

    }


    matrix = new double*[rows];


    for ( int j = 0; j < cols; j++){


        matrix[rows] = new double[cols];


    }

    //create the matrix


    for( int j=0; j < cols; j++ ){

        for (int i =0; i < rows; i++){

            matrix[i][j] = vector[cols*i + j];


        }
    }

}

