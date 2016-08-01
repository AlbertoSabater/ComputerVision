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

double getMean(vector<double> descriptors);
double getVariance(vector<double> descriptors, double mean);
double normVariance(vector<double> descriptors, double mean, double variance);


/**
 * Get the descriptors from the given image
 */
void getDescriptorsLearning(Mat src, int index, vector<double>& perimeter,
		vector<double>& hu0, vector<double>& hu1, vector<double>& hu2) {
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

	/// Draw contours and and store descriptors
	Mat drawing(dst.size(), CV_8UC3);
	double humu[7];
	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) > 53) {
			Moments mu = moments(contours[i], false);
			HuMoments(mu, humu);

			drawContours(drawing, contours, i, Scalar(0, 0, 200), 2, 8,
					hierarchy, 0, Point());

			imshow("Contorno", drawing);


			perimeter[index] = arcLength(contours[i], true);
			hu0[index] = humu[0];
			hu1[index] = humu[1];
			hu2[index] = humu[2];

			index++;
		}
	}
}


/**
 * Calculate the variance and mean from each descriptors and store them in the file fs
 */
void calculateDescriptors(FileStorage fs, vector<double> perimeter,
		vector<double> hu0, vector<double> hu1, vector<double> hu2, std::string currentObject) {

	double meanPerimeter = getMean(perimeter);
	double variancePerimeter = getVariance(perimeter, meanPerimeter);

	double meanhu0 = getMean(hu0);
	double variancehu0 = getVariance(hu0, meanhu0);

	double meanhu1 = getMean(hu1);
	double variancehu1 = getVariance(hu1, meanhu1);

	double meanhu2 = getMean(hu2);
	double variancehu2 = getVariance(hu2, meanhu2);

	double normP = normVariance(perimeter, meanPerimeter, variancePerimeter);
	double normhu0 = normVariance(hu0, meanhu0, variancehu0);
	double normhu1 = normVariance(hu1, meanhu1, variancehu1);
	double normhu2 = normVariance(hu2, meanhu2, variancehu2);


	fs << currentObject + std::string("_meanPerimeter") << meanPerimeter;
	fs << currentObject + std::string("_meanhu0") << meanhu0;
	fs << currentObject + std::string("_meanhu1") << meanhu1;
	fs << currentObject + std::string("_meanhu2") << meanhu2;

	fs << currentObject + std::string("_varianceNormPerimeter") << normP;
	fs << currentObject + std::string("_varianceNormhu0") << normhu0;
	fs << currentObject + std::string("_varianceNormhu1") << normhu1;
	fs << currentObject + std::string("_varianceNormhu2") << normhu2;

	fs << currentObject + std::string("_variancePerimeter") << variancePerimeter;
	fs << currentObject + std::string("_variancehu0") << variancehu0;
	fs << currentObject + std::string("_variancehu1") << variancehu1;
	fs << currentObject + std::string("_variancehu2") << variancehu2;

}


/**
 * Return the mean from the given descriptors
 */
double getMean(vector<double> descriptors) {
	double mean = 0.0;
	int size = descriptors.size();
	for (int i = 0; i < size; i++) {
		mean += descriptors[i];
	}
	return (mean / size);
}

/**
 * Return the variance from the given descriptors
 */
double getVariance(vector<double> descriptors, double mean) {
	double var = 0.0;
	int size = descriptors.size();
	for (int i = 0; i < size; i++) {
		var += pow((descriptors[i] - mean), 2.0);
	}
	return var / (size - 1);
}

/**
 * Return the normalized variance
 */
double normVariance(vector<double> descriptors, double mean, double variance) {
	//return (pow(descriptors[0] - mean, 2)/descriptors.size()) + (descriptors.size()-1)*variance/descriptors.size();
	return (pow(0.01*mean, 2)/descriptors.size()) + (descriptors.size()-1)*variance/descriptors.size();
}
