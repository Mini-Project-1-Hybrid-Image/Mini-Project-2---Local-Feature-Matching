import numpy as np
import cv2
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, gaussian
import matplotlib as plt
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage, misc
from skimage.feature import peak_local_max,corner_peaks
from scipy import signal



def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    def Cal_Ix_Iy(image:np.array,ksize):

        img=np.copy(image)
        img=skimage.img_as_float32(img)
        Ix = (cv2.Sobel((img), cv2.CV_8U, 1, 0, ksize))
        Iy = (cv2.Sobel((img), cv2.CV_8U, 0, 1, ksize))
        return Ix,Iy

    def Get_R (Ix,Iy,ksize,k,width) :
    ##This function apply the window (the gaussian filter to the derivatives then calculate R value)
        I_x=cv2.GaussianBlur(Ix**2, (ksize, ksize),0)
        I_y=cv2.GaussianBlur(Iy**2, (ksize, ksize), 0)
        H_xy=cv2.GaussianBlur(Ix*Iy, (ksize, ksize),0)
        R=((I_x*I_y-np.square(H_xy))-(k*np.square(I_x+I_y)))

        return R

  
    
    sobel_ksize=5
    if (feature_width%2==0):
        gaussian_ksize=feature_width+1
    else :
         gaussian_ksize=feature_width+1
    k=.0001

    Ix,Iy=Cal_Ix_Iy(image,sobel_ksize)
 
    R=Get_R(Ix,Iy,gaussian_ksize,k,16)
    
   

    print("Getting_corners!!")
    coordinates = ndimage.maximum_filter(R, size=3) ##I used this function to get the local maxima from surrounding points 
    R=coordinates
    x=[]
    y=[]
    for i in range(R.shape[0]):
        for j in range (R.shape[1]):
            if R[i][j]!=0:
                
                x.append(j)
                y.append(i)
                
   
    
    


    return np.array(x),np.array(y)
 
def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    ##Intiliazing the needed arrays ,vectors 
    feature=np.zeros((16,8))
    all_features=np.zeros((len(x),128))

    x=np.round(x).astype(int)
    y=np.round(y).astype(int)
    
    
    ## gaussian to the image for smoothing and noise reduction 
    gauss_img=gaussian(image)
    
    ##dx ,dy
    #dx = cv2.Sobel(gauss_img, cv2.CV_8U, 1, 0 , 5)                 
    #dy = cv2.Sobel(gauss_img, cv2.CV_8U, 0, 1 , 5) 

    #dx = scharr_v(gauss_img)
    #dy = scharr_h(gauss_img)

    dy, dx = np.gradient(gauss_img)
    
    ## Grad , Orientaion in degrees
    grad = np.sqrt(np.square(dx) + np.square(dy)) 
    img_oriention=np.arctan2(dy, dx)*(180/np.pi) 
    img_oriention[img_oriention<0]+=(2*np.pi)
    
    ###Quantiz the gradient into 8 directions 
    for i in range (0,8):
        x_,y_=np.where(((img_oriention>=(i*45)) & (img_oriention<((i+1)*45))))
        img_oriention[x_,y_]=int(i+1)
    
        
    
    ##
    window_size=np.round(feature_width/4 ).astype(int)
    
    ##create 16*16 grid around each point
    for i in range (len(x)):
        temp_orient=img_oriention[y[i]-8:y[i]+8,x[i]-8:x[i]+8] 
        temp_grad=grad[y[i]-8:y[i]+8,x[i]-8:x[i]+8]
        
        x_dir=0
        y_dir=0
        
        ##divide the grid into 4*4 blocks (16->4*4 blocks)
        for j in range (16):
            
            window=temp_orient[y_dir:y_dir+window_size,x_dir:x_dir+window_size]
            grad_win=temp_grad[y_dir:y_dir+window_size,x_dir:x_dir+window_size]
            
            ##histogram for each orientian window weighted by its magnitude
            hist=np.histogram((window),bins=8,range=(1, 9),weights=(grad_win))
            feature[j,:]=np.array(hist[0])
            y_dir+=4
            
            #every for iteratin :
            if y_dir==feature_width:
                x_dir+=4
                y_dir=0
                
        ##reshaping to 1*128 vector 
        feature_reshaped=feature.reshape(1,-1)  
        all_features[i,:]=feature_reshaped

    #Noormalize to 1 
    norm = np.linalg.norm(all_features, axis=1).reshape(-1, 1)
    norm[norm==0]=.01   ##to avoid division by 0
    all_features=all_features/norm

    ##Solving illumniatin problem ((cut all the values above .2))
    all_features[all_features>.2]=.2

    ##Normalize again 
    norm2 = np.linalg.norm(all_features, axis=1).reshape(-1, 1)
    norm2[norm2==0]=.01
    features=all_features/norm2
    


    return (features**.75)


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.
    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.
    For extra credit you can implement spatial verification of matches.
    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.
    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).
    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2
    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # lists to append to 
    matches = []
    confidences = []
    
    # Loop over the number of features in the first image
    for i in range(im1_features.shape[0]):
         ##Get the eculdine distance between each fature in image 1 and all features in image 2 

        eculdine_dist = np.sqrt(((im1_features[i,:]-im2_features)**2).sum(axis = 1))


        # sorting the distances (smaller -> bigger ) to get the smallest two imags 
        sortedt_dist = np.argsort(eculdine_dist)
        

        ##Applying the ratio test between the two smallest distnaces 
        ## if the ratio between them is less than .9 then consider it a good match 
        ## append to matches list  
        ##append the confidence 
        if (eculdine_dist[sortedt_dist[0]]/ eculdine_dist[sortedt_dist[1]])<.93:

         ### i is the index of image1 feature , image2 feature index : sortedt_dist[0]

            matches.append([i, sortedt_dist[0]])
            confidences.append((1.0  - eculdine_dist[sortedt_dist[0]]/eculdine_dist[sortedt_dist[1]])*100)
  
     

    return np.asarray(matches), np.asarray(confidences)
   