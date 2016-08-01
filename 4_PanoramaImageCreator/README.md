# Panorama image creator

This program mixes two or more images to create a panorama image. 

Feature points from each image are calculated with the Surf method and features from two images are matched. From this matchings, homography is calculated with the RANSAC method and the images are transformed and mixed to create a bigger panorama image.

![Alt text](images/matchings.png?raw=true "Matchings")
![Alt text](images/panorama.png?raw=true "Panorama image")

<br />
This program also works with photos taken from a WebCam.

![Alt text](images/realtime.png?raw=true "Photos from WebCam")











This detection has been implemented with direct and indirect vote.

### Direct vote
The original image has been processed with the Sobel operator to get the horizontal and vertical gradients. These gradients are used to get the magnitude function and the phase function.

The horizon line is placed in the middle of the image. To detect the vanishing point, each pixel who exceed a certain threshold votes to a point in the horizon line according to his position and gradient direction.

![Alt text](images/direct.png?raw=true "Direct vote")

### Indirect vote
The original image is processed with the Canny operator. After that, HoughLines function is used to get the straight lines in the image. These lines are intersected each other, and the point in the image with more intersections is the vanishing point.

![Alt text](images/indirect.png?raw=true "Indirect vote")

<br />
	* This program also works with videos from WebCam in real-time.
