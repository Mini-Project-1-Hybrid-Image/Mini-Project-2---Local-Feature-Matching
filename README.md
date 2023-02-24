#Introduction

Extracting features and image matching helps in image recognition. Thus in this project,
we apply Harris-Corner algorithm and SIFT to match different images.

#Harris corner[get_interest_points()]

Algorithm
● Using Sobel, we calculate the x and y directions derivatives.
● After that, we applied a gaussian filter to the derivatives to calculate the R value.
1. After that we used this
R=((I_x*I_y-np.square(H_xy))-(k*np.square(I_x+I_y))) to calculate the R
value.

● Then we use the maximum_filter function to get the local maximum from the
surrounding points.
● Then we do the non-maximum suppression by looping all over R and we do the
following,
1. If R[i][j]!=0, we append the indexes i, j in y, x directions.

Output
● The required indexes appended in the y, x directions.
