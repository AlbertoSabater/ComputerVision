#ifndef SRC_RECOGNITION_H_
#define SRC_RECOGNITION_H_

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


void fillValues(FileStorage fs, std::string objects[]);

void getDescriptorsRecognition(Mat src, std::string objects[]);


#endif /* SRC_RECOGNITION_H_ */
