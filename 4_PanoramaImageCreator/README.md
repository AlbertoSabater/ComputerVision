# Panorama image creator

This program mixes two or more images to create a panorama image. 

Feature points from each image are calculated with the Surf method and features from two images are matched. 

![Alt text](images/matchings.png?raw=true "Matchings")

From this matchings, homography is calculated with the RANSAC method and the images are transformed and mixed to create a bigger panorama image.

![Alt text](images/panorama.png?raw=true "Panorama image")

<br />
This program also works with photos taken from a WebCam.

![Alt text](images/realtime.png?raw=true "Photos from WebCam")