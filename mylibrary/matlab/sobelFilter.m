


%%
clear;
clc;

A = imread('driveexterior.png');
%A = imread('maxresdefault.jpg');



A_grey = rgb2gray(A);

imshow(A_grey);

G_x = [ -1 0 1; -2  0  2; -1  0  1];
G_y = [ 1 2 1; 0  0  0; -1  -2  -1];

%filter size is 3x3 

M_x = convolve(A_grey, G_x);
M_y = convolve(A_grey, G_y);

%M_output = size(M_x);

M = M_x.^2 + M_y.^2;

M = sqrt(M);

%hold off
imshow(M_x, [50, 150])


%%
%%Sobel filter

function [M] = convolve(A,F)

%for the sake of simplcity take image with size greatrer than 3x3 atleas
M = zeros(size(A,1), size(A,2));
size(M);

 for i= 2:(size(M,1)-1)
     for j = 2:(size(M,2)-1)
        
        M(i,j) = A(i-1,j-1)*F(1,1) + A(i-1,j)*F(1,2) +A(i-1,j+1)*F(1,3) + ... 
                 A(i,j-1)*F(2,1) + A(i,j)*F(2,2) + A(i,j+1)*F(2,3) + ...
                  A(i+1,j-1)*F(3,1) + A(i+1,j)*F(3,2) + A(i+1,j+1)*F(3,3);
           

     end
 end

end

%%


