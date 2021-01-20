Optical flow
------------
Optical flow or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual 
scene caused by the relative motion between an observer (an eye or a camera) and the scene.


1. Optical Flow Estimation
--------------------------
Implements Lucas-Kanade optical flow estimation, and test it for the two-frame data sets provided viz basketball, grove, and teddy.


2. Gaussian Pyramid
-------------------
Implements Lucas-Kanade optical flow estimation algorithm in a multi-resolution Gaussian pyramid
framework. After experimentally optimizing number of levels for Gaussian pyramid, local window size, and Gaussian width, I have used the same data sets (basketball, grove, and teddy) to find optical flows, visually compared my results with the previous step where I don’t use Gaussian pyramid.