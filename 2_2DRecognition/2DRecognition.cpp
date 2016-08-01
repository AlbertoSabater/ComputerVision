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
#include "learning.h"
#include "recognition.h"

using namespace cv;

void learning();
void recognition();

std::string objects[5] = { "circulo", "rectangulo", "rueda", "triangulo",
		"vagon" };

int main(int argc, char *argv[]) {
	std::cout
			<< "Pulse la tecla 'l' para comenzar entrenamiento o 'r' para empezar reconocimiento"
			<< std::endl;
	char key = std::cin.get();

	if (key == 'l') {
		learning();
	} else if (key == 'r') {
		recognition();
	} else {
		std::cout << "Pulse la tecla 'l' o 'r'" << std::endl;
	}
}

/**
 * For each training image, calculate their descriptors and store his variance
 * and mean in a file
 */
void learning() {
	FileStorage fs = FileStorage(std::string("descriptors.yml"),
			FileStorage::WRITE);

	std::string currentObject = "";
	vector<double> perimeter;
	vector<double> hu0, hu1, hu2;

	for (int f = 0; f < 5; f++) {		// For each object type
		std::cout << "-----------------------------" << std::endl;
		std::cout << "    Learning " << objects[f] << std::endl;
		std::cout << "-----------------------------" << std::endl;

		currentObject = objects[f];

		perimeter = vector<double>(5);
		hu0 = vector<double>(5);
		hu1 = vector<double>(5);
		hu2 = vector<double>(5);

		for (int i = 0; i < 5; i++) {
			string num = static_cast<std::ostringstream*>(&(std::ostringstream()
					<< i + 1))->str();

			std::string path = std::string("data/") + objects[f] + num
					+ std::string(".pgm");

			std::cout << path << std::endl;

			Mat src = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
			imshow("Image", src);
			getDescriptorsLearning(src, i, perimeter, hu0, hu1, hu2);
			waitKey(0); // wait for key press
		}
		calculateDescriptors(fs, perimeter, hu0, hu1, hu2, currentObject);
		std::cout
				<< "=============================================================================="
				<< std::endl;
	}

	fs.release();
}

/**
 * For each test file, get their objects and check his class
 */
void recognition() {
	FileStorage fs = FileStorage(std::string("descriptors.yml"),
			FileStorage::READ);
	fillValues(fs, objects);

	for (int i = 0; i < 3; i++) {

		string num = static_cast<std::ostringstream*>(&(std::ostringstream()
				<< i + 1))->str();

		string path = std::string("data/reco") + num + std::string(".pgm");
		std::cout << "-----------------------------" << std::endl;
		std::cout << "-  Fichero: " << path << "  -" << std::endl;
		std::cout << "-----------------------------" << std::endl;

		Mat src = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Image", src);

		getDescriptorsRecognition(src, objects);

		std::cout << "=============================" << std::endl;

		waitKey(0); // wait for key press
	}

	fs.release();
}
