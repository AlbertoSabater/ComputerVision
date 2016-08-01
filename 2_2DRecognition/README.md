# 2D Recognition

This program recognizes plain objects from samples previously learned.

To do that, first of all, all samples are thresholded using the Otsu's method to detect their contours. For the learning phase, descriptor parameters (area and invariant moments) are obtained from this contours and his mean and variance are stored for each kind of plain object.

To recognize a new object, is necessary to get his descriptors and calculate the Mahalanobis distance to each kind of object and for each descriptor. A new object belong to a kind of object if the sum of all his distances passes the chi-squar test with 4 degrees of freedom.

Variance has been regularized according the number of samples.

![asdasdasd](images/processing.png?raw=true "Image processing and contour detection")