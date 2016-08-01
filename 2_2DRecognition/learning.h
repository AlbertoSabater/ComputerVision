#ifndef LEARNING_H
#define LEARNING_H

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


void calculateDescriptors(FileStorage fs, vector<double> perimeter,
		vector<double> hu0, vector<double> hu1, vector<double> hu2, std::string currentObject);

void getDescriptorsLearning(Mat src, int index, vector<double>& perimeter,
		vector<double>& hu0, vector<double>& hu1, vector<double>& hu2);



#endif
