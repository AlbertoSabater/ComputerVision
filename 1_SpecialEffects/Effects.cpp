#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>

cv::VideoCapture TheVideoCapturer;
cv::Mat img;
cv::Mat img2;
cv::Mat map_x;
cv::Mat map_y;
cv::Mat labels;
cv::Mat centers;

cv::Mat background;
cv::Mat newBackground;
char* background_name = "underthesea2.png";

cv::Mat hsv_alien;
cv::Mat ycc_alien;


double contrast = 0.5;
int brightness = 1;

int colorChannel = 2;
double numColorsOrigin = 64;

using namespace cv;

void poster();
void posterKmeans();
void kmeansClustering();
void alien();
void distortion(double k);
void update_map(double k);
void changeContrastRGB();
void changeContrastHSV();
void showHistogram(char* text, Mat& img);
void changeBackground();
void equalizeHistogram();

int main(int argc, char *argv[]) {
	char key = 0;

	int numSnapshot = 0;
	std::string snapshotFilename = "0";

	std::cout << "Select one of this actions:" << std::endl;
	std::cout << " 1. Change contrast with HSV" << std::endl;
	std::cout << " 2. Change contrast with BGR" << std::endl;
	std::cout << " 3. Alien (r/g/b)" << std::endl;
	std::cout << " 4. Posterize" << std::endl;
	std::cout << " 5. Posterize with clustering" << std::endl;
	std::cout << " 6. Barrel distortion" << std::endl;
	std::cout << " 7. Cushion distortion" << std::endl;
	std::cout << " 8. Change background" << std::endl;
	std::cout << " 9. Equalize histogram" << std::endl;
	std::cout << std::endl;

	std::cout << "Press 's' to take snapshots" << std::endl;
	std::cout << "Press 'Esc' to exit" << std::endl;
	std::cout << std::endl;


	TheVideoCapturer.open(1);

	if (!TheVideoCapturer.isOpened()) {
		TheVideoCapturer.open(0);
		if (!TheVideoCapturer.isOpened()) {
			std::cerr << "Could not open video" << std::endl;
			return -1;
		}
	}

	while (key != 27 && TheVideoCapturer.grab()) {
		if (TheVideoCapturer.retrieve(img)) {

			imshow("Original image", img);
			if (key == 49) {	// 1
				changeContrastHSV();
			}
			else if (key == 50) {	// 2
				changeContrastRGB();
			}
			else if (key == 51) {	// 3
				alien();
			}
			else if (key == 52) {	// 4
				poster();
			}
			else if (key == 53) {	// 5
				posterKmeans();
			}
			else if (key == 54) {	// 6
				distortion(10e-6);
			}
			else if (key == 55) {	// 7
				distortion(-20e-7);
			}
			else if (key == 56) {	// 8
				changeBackground();
			}
			else if (key == 57) {	// 9
				equalizeHistogram();
			}

			char aux = cv::waitKey(20);
			if (aux != 'ÿ') {
				if ((aux < 58 && aux > 48) || aux == 27) {
					if (key != aux) {
						if (aux == 54 || aux == 55) {		// Empty maps for distortions
							map_x = Mat_<int>(0,0);
							map_y = Mat_<int>(0,0);
						}
						else if (aux == 49 || aux == 50) {		// Get contrast and brightness
							std::cout << "=================================================" << std::endl;
							std::cout << "Contrast: " << std::endl;
							std::cin >> contrast;
							std::cout << "Brightness: " << std::endl;
							std::cin >> brightness;
							std::cout << "=================================================" << std::endl;
						}
						else if (aux == 52) {		// Get numColors to posterize
							std::cout << "=================================================" << std::endl;
							std::cout << "Insert number of colors to posterize (8, 27, 64, etc.)" << std::endl;
							std::cin >> numColorsOrigin;
							std::cout << "=================================================" << std::endl;
						}
						else if (aux == 53) {
							centers = Mat_<int>(0,0);
						}
						else if (aux == 56) {		// changeBackground
							background = img.clone();
							newBackground = imread(background_name, 1 );
						}
					}

					img2 = Mat_<int>(0,0);
					key = aux;
					destroyAllWindows();
				}
				// Channge alien oolor channel
				else if (aux == 114) {	// r
					colorChannel = 2;
				}
				else if (aux == 103) {	// g
					colorChannel = 1;
				}
				else if (aux == 98) {	// b
					colorChannel = 0;
				}
				else if (aux == 115) {

					if (!img.empty()) {
						cv::imwrite(snapshotFilename + ".png", img);
					}
					cv::imwrite(snapshotFilename + ".png", img);
					numSnapshot++;
					snapshotFilename =
							static_cast<std::ostringstream*>(&(std::ostringstream()
									<< numSnapshot))->str();
				}
			}


		}
	}

	//std::string s;
	for (int photo = 0; photo < numSnapshot; photo = photo + 1) {
		cv::Mat image;
		std::stringstream s;
		s << photo << ".png";
		//std::cout << s;
		image = cv::imread(s.str(), 1);
		cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
		cv::imshow("Display Image", image);
		cv::waitKey(0);
	}

}


//http://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
/**
 * Equalize histogram in YCrCb and convert to BGR
 */
void equalizeHistogram() {

	cvtColor(img,img2,CV_BGR2YCrCb);
	vector<Mat> channels;
	split(img2,channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels,img2);
	cvtColor(img2,img2,CV_YCrCb2BGR);

	imshow("Histogram equalized", img2);
	showHistogram("Original", img);
	showHistogram("Updated", img2);

}


/**
 * Changes the image background with the specified image
 * The first frame will me the old background
 */
void changeBackground() {

	img2 = img.clone();

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			// Euclidean distance
			double dist = sqrt(pow(img2.at<Vec3b>(y, x)[0] - background.at<Vec3b>(y, x)[0],2)
					+ pow(img2.at<Vec3b>(y, x)[1] - background.at<Vec3b>(y, x)[1],2)
					+ pow(img2.at<Vec3b>(y, x)[2] - background.at<Vec3b>(y, x)[2],2));

			if (dist < 60) {	// Change background
				img2.at<Vec3b>(y, x) = newBackground.at<Vec3b>(y, x);
			}
		}
	}

	imshow("Under the sea", img2);

}


/**
 * Posterize the image with the specified number of colors
 */
void poster() {
	Mat img2 = img.clone();
	GaussianBlur(img2, img2, Size(7,7), 1, 1);

	double numColors = cbrt(numColorsOrigin);

	std::vector<int> colors((int) numColors);

	/*
	colors[0] = 0;
	colors[numColors - 1] = 255;
	double step = 255 / (numColors - 1);
	double current = step;
	for (int i = 1; i < numColors - 1; i++) {		// Create new color scale
		colors[i] = current;
		current += step;
	}
	*/

	double step = 255 / (numColors+1);
	double current = step;
	for (int i = 0; i < numColors; i++) {		// Create new color scale
		colors[i] = current;
		current += step;
	}

	for (int j = 0; j < img.rows; j++) {		// Rescale all colors in all channels
		uchar* data = img2.ptr<uchar>(j);

		for (int i = 0; i < img.cols * img.channels(); i++) {
			double index = data[i] * numColors / 255;
			int index2 = index;
			if (index - index2 > 0.5) { ++index2; }
			if (index2 == numColors) { --index2; }
			data[i] = colors[index2];
		}
	}

	imshow("Original image", img);
	imshow("Posterized image", img2);

}


/**
 * Posterize the image with the specified number of colors using kmeans
 */
void posterKmeans() {
	if (centers.rows == 0) {
		std::cout << "Kmeans started" << std::endl;
		kmeansClustering();
		std::cout << "Kmeans finished" << std::endl;
	}


	Mat img2(img.size(), img.type());
	/*for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * img.rows, 0);
			img2.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			img2.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			img2.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}*/

	/*for (int c = 0; c < centers.rows; c++) {
		std::cout << "Center " <<  c << "  " << centers.at<Vec3b>(c) << std::endl;
	}*/

	double minDistance = std::numeric_limits<double>::infinity();
	int index = 0;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			for (int c = 0; c < centers.rows; c++) {
				double dist = pow(img.at<Vec3b>(y, x)[0] - centers.at<Vec3b>(c)[0],2)
						+ pow(img.at<Vec3b>(y, x)[1] - centers.at<Vec3b>(c)[1],2)
						+ pow(img.at<Vec3b>(y, x)[2] - centers.at<Vec3b>(c)[2],2);

				if (dist < minDistance) {
					minDistance = dist;
					index = c;
				}
			}
			img2.at<Vec3b>(y, x) = centers.at<Vec3b>(index);
			minDistance = std::numeric_limits<double>::infinity();
		}
	}

	imshow("Posterized image with kmeans", img2);
}


/**
 * Generate the centroids
 */
void kmeansClustering() {

	std::cout << "in kmeans" << "\n" << std::endl;

	Mat samples(img.rows * img.cols, 3, CV_32F);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			for (int z = 0; z < 3; z++) {
				samples.at<float>(y + x * img.rows, z) = img.at<Vec3b>(y, x)[z];
			}
		}
	}

	int clusterCount = 64;
	int attempts = 3;
	kmeans(samples, clusterCount, labels,
			TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
			attempts, KMEANS_PP_CENTERS, centers);

}


/**
 * Changes the skin color in the specified RGB channel
 */
void alien() {

	Scalar hsv_min = Scalar(4, 10, 20);
	Scalar hsv_max = Scalar(20, 150, 255);

	cvtColor(img, hsv_alien, CV_BGR2HSV);
	//GaussianBlur(hsv_alien, hsv_alien, Size(7,7), 1, 1);
	normalize(hsv_alien, hsv_alien, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
	inRange(hsv_alien, hsv_min, hsv_max, hsv_alien);

	Scalar ycc_min = Scalar(0,133,75);
	Scalar ycc_max = Scalar(255,173,127);
	cv::cvtColor(img,ycc_alien,cv::COLOR_BGR2YCrCb);
	//GaussianBlur(ycc_alien, ycc_alien, Size(7,7), 1, 1);
	normalize(ycc_alien, ycc_alien, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
	cv::inRange(ycc_alien,ycc_min,ycc_max,ycc_alien);

	img2 = img.clone();
	for (int j = 0; j < hsv_alien.rows; j++) {
		for (int i = 0; i < hsv_alien.cols; i++) {
			if (ycc_alien.at<uchar>(j, i) == 255 && hsv_alien.at<uchar>(j, i) == 255) {
				if (img2.at<cv::Vec3b>(j, i)[colorChannel] * 1.5 > 255) {
					img2.at<cv::Vec3b>(j, i)[colorChannel] = 255;
				} else {
					img2.at<cv::Vec3b>(j, i)[colorChannel] *= 1.5;
				}
				/*img2.at<cv::Vec3b>(j, i)[colorChannel] = 255;
				for (int q = 0; q < 2; q++) {
					if (colorChannel != q) {
						img2.at<cv::Vec3b>(j, i)[q] *= 0.25;
					}
				}*/
			}
		}
	}

	imshow("Skin hsv", hsv_alien);
	imshow("Skin ycc", ycc_alien);
	imshow("Colored skin", img2);
}


/**
 * Distorts the image with the specified k. Barrel or cushion distortion
 */
void distortion(double k) {

	if (map_x.rows == 0) {		// Generate maps to distortion
		update_map(k);
	}

	// Distorts the image
	remap(img, img2, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	cv::imshow("Original image", img);
	cv::imshow("Updated image", img2);
}

void update_map(double k) {

	img2.create(img.size(), img.type());
	map_x.create(img.size(), CV_32FC1);
	map_y.create(img.size(), CV_32FC1);

	int centerX = img.cols / 2;
	int centerY = img.rows / 2;

	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			double x = (double) (i - centerX);
			double y = (double) (j - centerY);
			double r = sqrt(x * x + y * y);

			map_x.at<float>(j, i) = x * (1 + k * pow(r, 2)) + centerX;
			map_y.at<float>(j, i) = y * (1 + k * pow(r, 2)) + centerY;
		}
	}
}


/**
 * Change contrast and brightness in RBG color space
 */
void changeContrastRGB() {
	Mat histogram, histogram2;
	img2 = img.clone();


	for (int y = 0; y < img.rows; y++) {
		uchar* data = img2.ptr<uchar>(y); // puntero a la fila i
		for (int x = 0; x < img.cols * img.channels(); x++) {
			for (int c = 0; c < 3; c++) {
				data[x] = saturate_cast<uchar>(contrast * (data[x]) + brightness);
			}
		}
	}

	cv::imshow("Original image", img);
	cv::imshow("Updated image BGR", img2);
	showHistogram("Histogram Original BGR ", img);
	showHistogram("Histogram Updated BGR ", img2);
}

/**
 * Change contrast and brightness in HSV color space
 */
void changeContrastHSV() {
	Mat histogram, histogram2;
	cvtColor(img, img2, CV_BGR2HLS_FULL);

	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for (int y = 0; y < img2.rows; y++) {
		for (int x = 0; x < img2.cols; x++) {
			img2.at<cv::Vec3b>(y,x)[2] = saturate_cast<uchar>(contrast * img2.at<cv::Vec3b>(y,x)[2]);
			img2.at<cv::Vec3b>(y,x)[1] = saturate_cast<uchar>(brightness * img2.at<cv::Vec3b>(y,x)[1]);
		}
	}

	cvtColor(img2, img2, CV_HLS2BGR_FULL);	// Convert to grayscale

	cv::imshow("Original image", img);
	cv::imshow("Updated image HSL", img2);
	//showHistogram("Histogram Original HSV ", img);
	//showHistogram("Histogram Updated HSV", img2);
}

//http://opencv-code.com/tutorials/drawing-histograms-in-opencv/
/**
 * Show the histogram in each channel
 */
void showHistogram(char* text, Mat& img) {
	int bins = 256;             // number of bins
	int nc = img.channels();    // number of channels

	vector<Mat> hist(nc);       // histogram arrays

	// Initalize histogram arrays
	for (int i = 0; i < hist.size(); i++)
		hist[i] = Mat::zeros(1, bins, CV_32SC1);

	// Calculate the histogram of the image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			for (int k = 0; k < nc; k++) {
				uchar val =
						nc == 1 ? img.at<uchar>(i, j) : img.at<Vec3b>(i, j)[k];
				hist[k].at<int>(val) += 1;
			}
		}
	}

	// For each histogram arrays, obtain the maximum (peak) value
	// Needed to normalize the display later
	int hmax[3] = { 0, 0, 0 };
	for (int i = 0; i < nc; i++) {
		for (int j = 0; j < bins - 1; j++)
			hmax[i] =
					hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
	}

	char* wname[3] = { "blue", "green", "red" };
	Scalar colors[3] =
			{ Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };

	vector<Mat> canvas(nc);

	// Display each histogram in a canvas
	for (int i = 0; i < nc; i++) {
		canvas[i] = Mat::ones(100, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins - 1; j++) {
			line(canvas[i], Point(j, rows),
					Point(j, rows - (hist[i].at<int>(j) * rows / hmax[i])),
					nc == 1 ? Scalar(200, 200, 200) : colors[i], 1, 8, 0);
		}

		if (nc == 1) {
			//imshow(text + " BN", canvas[i]);
		}

		std::string final_text = std::string(text) + std::string(wname[i]);
		imshow(nc == 1 ? "value" : final_text, canvas[i]);
		//std::cout << text + nc + '\n';
		//imshow(text + ' ' + nc, canvas[i]);
	}
}

