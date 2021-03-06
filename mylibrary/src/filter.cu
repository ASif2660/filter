//
// Created by asif on 07.03.19.
//

#include "../include/filter.cuh"



void simple_edge_detector::edge_detector::set_output_eigen_matrix(MatrixXf &eigen_matrix) {


    _eigenMatrix = eigen_matrix;

}


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



void simple_edge_detector::vector_to_matrix(double* vector, MatrixXf& eigen_output){


    // this basically converts the vector to matrix of required size
/*
    if( size_of_vector !=  ROWS * COLS ){


        std::cout << " This is an illegal conversion !!!, Please check the total size, rows and cols " << std::endl;

    }



    matrix = new double*[rows];


    for ( int j = 0; j < cols; j++){


        matrix[rows] = new double[cols];


    }

    //create the matrix
*/

    for( int j=0; j < COLS; j++ ){

        for (int i =0; i < ROWS; i++){

            eigen_output(i,j) = vector[COLS*i + j];


        }
    }

}



__global__
void simple_edge_detector::edge_detector_gpu (double* vectorized_matrix, double* kernel_x, double* kernel_y, double* output_vector_matrix)

{

   int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    //actual size is rows = 1080 and columns 1307

    if(row < 768 && col < 1024) { //within the inside indices {

        output_vector_matrix[row*1307 + col] = vectorized_matrix[row*1307 + col -col -1]* kernel_x[0] +

                vectorized_matrix[row*1307 +col -col]*kernel_x[1] + vectorized_matrix[row*1307 +col -col+ 1]*kernel_x[2] +

                vectorized_matrix[row*1307+ col-1]* kernel_x[3] + vectorized_matrix[row*1307 +col]*kernel_x[4] +

                vectorized_matrix[row*1307+ col+ 1]*kernel_x[5] + vectorized_matrix[row*1307+ col + col -1 ]*kernel_x[6]+

                vectorized_matrix[row*1307 +col + col]* kernel_x[7]  + vectorized_matrix[row*1307 +col +col+1]*kernel_x[8];



        // need to figure out the jumps top and bottom

    }


/*
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

// indexing starts from zero
 if (row < 768 && col < 1024)
 output_vector_matrix[1024*row + col] = vectorized_matrix[1024*row + col];

// printf("Hello World from GPU!\n");

 //output_vector_matrix[index] =  6.0;
//  printf("The value of at %d is %f \n ", output_vector_matrix[i] );

*/
}

void simple_edge_detector::edge_detector::execute_kernels(double* global_vectorized_matrix, double* global_kernel_x, double* global_kernel_y,

                  double* global_output_matrix){


 //TODO: initialize global_output_matrix before passing it

 unsigned int threads_per_block = 128;

 dim3 GRID(threads_per_block,threads_per_block); // 64 by 64

 dim3 BLOCK(COLS/GRID.x +1 , ROWS/GRID.y + 1);

 cudaMalloc((void**)&device_vectorized_matrix, ROWS*COLS*sizeof(double));

 cudaMalloc((void**)&device_gpu_output, ROWS*COLS*sizeof(double));

 cudaMalloc((void**)&device_kernel_x, 9*sizeof(double));

 cudaMalloc((void**)&device_kernel_y, 9*sizeof(double)); // kernel size is hard coded for now, 3x3 is the size


 cudaMemcpy(device_vectorized_matrix, global_vectorized_matrix, ROWS*COLS*sizeof(double), cudaMemcpyHostToDevice);

 cudaMemcpy(device_kernel_x, global_kernel_x, 9*sizeof(double), cudaMemcpyHostToDevice);

 cudaMemcpy(device_kernel_y, global_kernel_y, 9*sizeof(double), cudaMemcpyHostToDevice);


  simple_edge_detector::edge_detector_gpu <<<GRID, BLOCK>>> (device_vectorized_matrix, device_kernel_x, device_kernel_y, device_gpu_output);

 //simple_edge_detector::edge_detector_gpu <<<GRID,BLOCK>>> (device_gpu_output);



 cudaMemcpy(global_output_matrix, device_gpu_output, ROWS*COLS*sizeof(double), cudaMemcpyDeviceToHost);

 //cudaMemcpy(global_output_matrix, device_vectorized_matrix, ROWS*COLS*sizeof(double), cudaMemcpyDeviceToHost);

 //initialize global_kernel_x, global_kernel_y and global matrix

 cudaFree(device_gpu_output);

 cudaFree(device_kernel_x);

 cudaFree(device_kernel_y);

 cudaFree(device_vectorized_matrix);



}