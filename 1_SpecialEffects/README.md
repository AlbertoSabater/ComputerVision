# Special Effects

This program modifies the video from Web Cam allowing the user to apply the following effects:

1. Contrast and brightness using the RGB or HSL color space <br />
![Alt text](images/brightness.png?raw=true "Modifying brightness")
![Alt text](images/contrast.png?raw=true "Modifying contrast")

2. Histogram equalization <br />
![Alt text](images/equalization.png?raw=true "Equalize histogram")

3. Alien mode <br />
This color transformation uses a mask in different color spaces to detect the color skin, and modify it.
![Alt text](images/alien.png?raw=true "Alien mode")

4. Posterize <br />
   * First version umbralizes all the RGB color space. <br />
![Alt text](images/poster.png?raw=true "Poster")

   * Second version uses K-means to clusterize the RGB image colors. <br />
![Alt text](images/poster-Kmeans.png?raw=true "Poster with K-means")

5. Barrel and pincushion distortion <br />
![Alt text](images/distortion.png?raw=true "Barrel and pincushion distortion")

6. Background effect <br />
![Alt text](images/background.png?raw=true "Background effect")
