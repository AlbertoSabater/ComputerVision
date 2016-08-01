#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>
#include <stdio.h>
#include <iomanip>
#include <locale>

using namespace cv;

double mahalanobisDistance(double value, double mean, double variance);
void checkClass(double perimeter, double hu0, double hu1, double hu2, std::string objects[]);

vector<double> perimeter;
vector<double> hu0;
vector<double> hu1;
vector<double> hu2;

vector<double> meanPerimeterValues(5);
vector<double> meanhu0(5);
vector<double> meanhu1(5);
vector<double> meanhu2(5);

vector<double> variancePerimeterValues(5);
vector<double> variancehu0(5);
vector<double> variancehu1(5);
vector<double> variancehu2(5);

vector<Mat> drawing(5);


/**
 * Get the descriptors from contours from the given image and check their classes
 */
void getDescriptorsRecognition(Mat src, std::string objects[]) {
	Mat dst;

	// Thresholding
	threshold(src, dst, 0, 255, CV_THRESH_OTSU);
	bitwise_not(dst, dst);		// Invert Image colors
	imshow("Image dst1", dst);

	// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,
			Point(0, 0));

	/// Draw contours and check class
	Moments mu;
	double humu[7];
	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) > 53) {
			mu = moments(contours[i], false);
			HuMoments(mu, humu);

			Mat draw = Mat::zeros(dst.size(), CV_8UC3);
			drawContours(draw, contours, i, Scalar(0, 0, 200), 2, 8,
					hierarchy, 0, Point());
			imshow("Contorno", draw);

			checkClass(arcLength(contours[i], true), humu[0], humu[1], humu[2], objects);

			waitKey(0); // wait for key press
		}
	}
}


/**
 * Read the descriptors's values stored in the file fs
 */
void fillValues(FileStorage fs, std::string objects[]) {
	for (int o = 0; o < 5; o++) {
		fs[objects[o] + std::string("_meanPerimeter")] >> meanPerimeterValues[o];

		fs[objects[o] + std::string("_meanhu0")] >> meanhu0[o];
		fs[objects[o] + std::string("_meanhu1")] >> meanhu1[o];
		fs[objects[o] + std::string("_meanhu2")] >> meanhu2[o];


		fs[objects[o] + std::string("_varianceNormPerimeter")] >> variancePerimeterValues[o];

		fs[objects[o] + std::string("_varianceNormhu0")] >> variancehu0[o];
		fs[objects[o] + std::string("_varianceNormhu1")] >> variancehu1[o];
		fs[objects[o] + std::string("_varianceNormhu2")] >> variancehu2[o];
	}
}


/**
 * Check the class with the given descriptors
 */
void checkClass(double perimeter, double hu0, double hu1, double hu2, std::string objects[]) {

	std::string possibleObject = "";
	double minDistance = HUGE_VAL;

	for (int o = 0; o < 5; o++) {				// For each object
		double distP = mahalanobisDistance(perimeter,
				meanPerimeterValues[o], variancePerimeterValues[o]);
		double disthu0 = mahalanobisDistance(hu0, meanhu0[o],
				variancehu0[o]);
		double disthu1 = mahalanobisDistance(hu1, meanhu1[o],
				variancehu1[o]);
		double disthu2 = mahalanobisDistance(hu2, meanhu2[o],
				variancehu2[o]);

		double totalDistance = distP + disthu0 + disthu1 + disthu2;

		//std::cout << "Total distance (" << objects[o] << "): " << totalDistance << std::endl;

		if (totalDistance <= 13.2767) {	// 0.01
			if (possibleObject == "") {
				possibleObject = objects[o];
			}
			else {
				possibleObject += " | " + objects[o];
			}
		}

		if (totalDistance < minDistance) {
			minDistance = totalDistance;
		}
	}

	if (possibleObject == "") {
		possibleObject = " --- ";
	}

	std::cout << "Clase reconocida: " << possibleObject << std::endl;
	std::cout << "Min distance: " << minDistance << std::endl;
}


/**
 * Return the Mahalanobis Distance between the given values
 */
double mahalanobisDistance(double value, double mean, double variance) {
	return pow(value - mean, 2) / variance;
}


