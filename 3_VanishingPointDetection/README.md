# Vanishing point detection

This detection has been implemented with direct and indirect vote.

### Direct vote
The original image has been processed with the Sobel operator to get the horizontal and vertical gradients. These gradients are used to get the magnitude function and the phase function.

The horizon line is placed in the middle of the image. To detect the vanishing point, each pixel who exceed a certain threshold votes to a point in the horizon line according to his position and gradient direction.

![Alt text](images/direct.png?raw=true "Direct vote")

### Indirect vote
The original image is processed with the Canny operator. After that, HoughLines function is used to get the straight lines in the image. These lines are intersected each other, and the point in the image with more intersections is the vanishing point.

![Alt text](images/indirect.png?raw=true "Indirect vote")


This program also works with videos from WebCam in real-time.













This program recognizes plain objects from samples previously learned.

To do that, first of all, all samples are thresholded using the Otsu's method to detect their contours. For the learning phase, descriptor parameters (area and invariant moments) are obtained from these contours and his mean and variance are stored for each kind of plain object.

To recognize a new object, is necessary to get his descriptors and calculate the Mahalanobis distance to each kind of object and for each descriptor. A new object belong to a kind of object if the sum of all his distances passes the chi-squar test with 4 degrees of freedom.

Variance has been regularized according the number of samples.

![Alt text](images/processing.png?raw=true "Image processing and contour detection")
<p align="center">
	Image processing and contour detection
</p>