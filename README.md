# Introduction

Extracting features and image matching helps in image recognition. Thus in this project,
we apply Harris-Corner algorithm and SIFT to match different images.

# Harris corner[get_interest_points()]
Harris corner[get_interest_points()]
Algorithm
Using Sobel, we calculate the x and y directions derivatives.
After that, we applied a gaussian filter to the derivatives to calculate the R value.
After that we used this R=((I_x*I_y-np.square(H_xy))-(k*np.square(I_x+I_y))) to calculate the R value.
Then we use the maximum_filter function to get the local maximum from the surrounding points.
Then we do the non-maximum suppression by looping all over R and we do the following,
If R[i][j]!=0, we append the indexes i, j in y, x directions. 
Output
The required indexes appended in the y, x directions.

![notre_1](https://user-images.githubusercontent.com/49596777/221068913-03d92b19-c882-4ed5-b6cc-a33b60752566.png)
![rushmore_2](https://user-images.githubusercontent.com/49596777/221068930-f29405a6-c7c4-479e-a2c1-d836e42749c0.png)
# Discussion
The threshold is the factor that determines how the algorithm will show the interest points, as by decreasing the threshold, the algorithm shows more interest points. By having more interest points, we can have more ability to catch an accepted feature, so the accuracy will increase. However, this may require more time and memory.  

# SIFT Descriptor
The SIFT descriptor is an algorithm used to extract features from an image. It is a technique that identifies and describes the unique characteristics of an image in a way that is invariant to rotation and illumination changes. The algorithm involves several steps: First, the gradient and orientation of the image are obtained. Then, a 1616 grid is created around each keypoint. The 1616 grid is divided into 4*4 blocks, and for each block, an oriented histogram is calculated. The histograms for all the 16 blocks are concatenated into a 128-dimensional vector that represents the feature. This process is repeated for all keypoints in the image. The resulting output is a matrix of shape (number of keypoints, 128) with each row representing a new feature. The algorithm is useful for object recognition, image stitching, and other computer vision applications.

# Features Matching 
The Features Matching involves several steps to match features between two images. For each feature in the first image, the algorithm calculates the Euclidean distance between that feature and all the features in the second image. The distances are sorted from smallest to largest. The algorithm then applies a ratio test between the two smallest distances to avoid incorrect matches. If the ratio is less than a certain threshold (usually set to 0.8), the match is considered valid, and it is appended to the matches list. The algorithm also calculates the confidence of the match and appends it to its corresponding list. The output of the algorithm is a vector containing all the matches and a vector containing the corresponding confidences of the matches. The Features Matching algorithm is useful for object recognition, image stitching, and other computer vision applications.


# Results 
    Notre_dame 
    Matches: 1546
    Accuracy on 50 most confident: 94%
    Accuracy on 100 most confident: 86%
    Accuracy on all matches: 52%
![CIE427_Fall2021_Mini-Project_Debug Entity_results_notre_dame_matches jpg](https://user-images.githubusercontent.com/49596777/221070047-63f15123-a794-4e07-8179-c504b59ebe6b.jpg)


      Mt.Rushmore 
      Matches: 221
      Accuracy on 50 most confident: 88%
      Accuracy on 100 most confident: 86%
      Accuracy on all matches: 76%
      
![CIE427_Fall2021_Mini-Project_Debug Entity_results_mt_rushmore_matches jpg](https://user-images.githubusercontent.com/49596777/221070158-4a90d711-5b81-409a-9066-2616a2d69a36.jpg)

      E-Gaudi :

      Matches: 97
      Accuracy on 50 most confident: 10%
      Accuracy on all matches: 17%
      
![CIE427_Fall2021_Mini-Project_Debug Entity_results_GaudiMatches jpg](https://user-images.githubusercontent.com/49596777/221070254-934b10ba-4fec-43d2-8744-21263615a450.jpg)

# 
From the above results :
        For notre dame pair ,the result is satisfying  and it's the highest result of all three results 
        For rushmore pair ,the accuracy is little less than notre accuracy.This may be due to the more hard corner to detect for harris algorithm 
        The E-Gauide pair result is not good but this due to the different scales of he two images as our implementation does not include scale invariance 













