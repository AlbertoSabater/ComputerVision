#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>

using namespace cv;

Mat sobel(Mat src);
Mat canny(Mat src);
Mat getLines(Mat src, Mat srcContours);
bool intersection(Point o1, Point p1, Point o2, Point p2, Point &r);
Point getCenterDirect(vector<Vec2f> lines, Mat &dst);
Point getCenterIndirect(vector<Vec2f> lines, Mat &dst);
int getClosestCenter(vector<Point> centers, Point p);
void printCross(Mat &src, Point p);
void getInfoLine(Vec2f line, Point &pt1, Point &pt2, double &rad_angle);
Point getVanishingPoint(Mat src);
void execute(Mat src);

Scalar red = Scalar(0, 0, 255);
Scalar green = Scalar(0, 255, 0);

Mat orientation;
Mat module;

int maxDistance = 15;
bool opt = true;
bool video = false;
bool sobel_method = false;

cv::VideoCapture TheVideoCapturer;


int main(int argc, char *argv[]) {

	if (!video) {
		for (int i = 1; i <= 3; i++) {
			string num = static_cast<std::ostringstream*>(&(std::ostringstream()
					<< i))->str();
			std::string path = std::string("ImagenesT2/pasillo") + num + std::string(".pgm");
			std::cout << path << std::endl;
			Mat src = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
			resize(src,src,Size(500,512));//resize image
			execute(src);
			waitKey(0); // wait for key press
		}
	}
	else {		// Enable video
		TheVideoCapturer.open(1);

		if (!TheVideoCapturer.isOpened()) {
			TheVideoCapturer.open(0);
			if (!TheVideoCapturer.isOpened()) {
				std::cerr << "Could not open video" << std::endl;
				return -1;
			}
		}

		char key = 0;

		Mat src;
		while (key != 27 && TheVideoCapturer.grab()) {
			if (TheVideoCapturer.retrieve(src)) {
				cvtColor(src, src, CV_RGB2GRAY);
				//imshow("Src", src);
				execute(src);
			}

			key = waitKey(50);
		}
	}

}

void execute(Mat src) {
	if (sobel_method) {
		Mat dst_sobel = sobel(src);

		Point res = getVanishingPoint(src);
		std::cout << res << std::endl;

		printCross(src, res);
		imshow("Original", src);
	}
	else {
		if (!video) {
			equalizeHist( src, src );
		}
		Mat dst_canny = canny(src);
		Mat lines = getLines(src, dst_canny);
		imshow("lines", lines);
	}
}


/*
 * Return contours from src
 */
Mat canny(Mat src) {

	int thresh = 1000;
	int max_thresh = 3000;

	Mat canny_output;

	/// Detect edges using canny
	Canny(src, canny_output, thresh, max_thresh, 5);

	return canny_output;
}

/*
 * Apply sobel operator to src, getting gradient and module
 */
Mat sobel(Mat src) {
	Mat grad;
	int ddepth = CV_16S;
	int kernel_size = 3;

	GaussianBlur(src, src, Size(kernel_size, kernel_size), 0, 0,
			BORDER_DEFAULT);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src, grad_x, ddepth, 1, 0, kernel_size);
	convertScaleAbs(grad_x, abs_grad_x);
	abs_grad_x.convertTo(abs_grad_x, CV_32F);
	//imshow("abs_grad_x", abs_grad_x);

	/// Gradient Y
	//Scharr( src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src, grad_y, ddepth, 0, 1, kernel_size);
	convertScaleAbs(grad_y, abs_grad_y);
	abs_grad_y.convertTo(abs_grad_y, CV_32F);
	//imshow("abs_grad_y", abs_grad_y);

	/*Mat gx = abs_grad_x.clone();
	Mat gy = abs_grad_y.clone();
	gx.convertTo(gx, CV_8U);
	gy.convertTo(gy, CV_8U);
	gx = gx/2 + 128;
	gy = gy/2 + 128;
	imshow("Gradiente horizontal", gx);
	imshow("Gradiente vertical", gy);*/

	module = abs_grad_x.clone();
	magnitude(abs_grad_x, abs_grad_y, module);
	module.convertTo(module, CV_8U);
	imshow("magnitude", module);

	orientation = abs_grad_x.clone();
	phase(abs_grad_x, abs_grad_y, orientation, false);

	/*Mat ori = orientation.clone();
	ori = ori*128/CV_PI;
	ori.convertTo(ori, CV_8U);
	imshow("orientation", ori);*/

	return grad;

}

/**
 * Return the vanishing point with direct vote from the module and orientation
 */
Point getVanishingPoint(Mat src) {

	int midImage = src.rows/2;
	Point mid1(0, midImage);
	Point mid2(src.cols, midImage);

	Mat trh;
	threshold(module, trh, 20, 255, 0);

	line(src, mid1, mid2, Scalar(0,0,0));

	vector<int> van(src.cols);

	for (int i=0; i<src.rows; i++) {
		for (int j=0; j<src.cols; j++) {
			if (trh.at<uchar>(i,j) == 255) {

				for (float k = -0.2; k <= 0.02; k+=0.02) {

					//float x = j - src.cols/2;
					//float y = src.rows/2 - i;
					float theta = orientation.at<float>(j,i) + k;
					//float rho = x*cos(theta) + y*sin(theta);
					float rho = j*cos(theta) + i*sin(theta);

					Point pt1, pt2;
					double a = cos(theta), b = sin(theta);
					double x0 = a * rho, y0 = b * rho;
					pt1.x = cvRound(x0 + 1000 * (-b));
					pt1.y = cvRound(y0 + 1000 * (a));
					pt2.x = cvRound(x0 - 1000 * (-b));
					pt2.y = cvRound(y0 - 1000 * (a));
					Point res;
					bool intersect = intersection(pt1, pt2, mid1, mid2, res);

					if (intersect && res.x < 500 && res.x > 0) {
						van[res.x] ++;
					}
				}

			}

		}
	}
	imshow("trh", trh);

	int max = 0;
	Point center;
	center.y = midImage;

	for (int i=1; i< src.cols-1; i++) {
		if (van[i] >= max) {
			max = van[i];
			center.x = i;
		}
	}

	return center;
}


Mat getLines(Mat src, Mat srcContours) {

	Mat dst;
	cvtColor(src, dst, cv::COLOR_GRAY2BGR);

	int line_width = 1;

	vector<Vec2f> lines;
	HoughLines(srcContours, lines, 1, CV_PI / 180, 100);

	/*for (size_t i = 0; lines.size() != 0 && i < lines.size(); i++) {
		Point pt1, pt2; double rad_angle;
		getInfoLine(lines[i], pt1, pt2, rad_angle);
		line(dst, pt1, pt2, green, line_width, CV_AA);
	}*/

	Point center;
	if (opt) {
		center = getCenterIndirect(lines, dst);
	}
	else {
		center = getCenterDirect(lines, dst);
	}
	printCross(dst, center);

	return dst;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point o1, Point p1, Point o2, Point p2, Point &r) {
	Point x = o2 - o1;
	Point d1 = p1 - o1;
	Point d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

/**
 * Calcule vanishing point from lines with direct vote
 */
Point getCenterDirect(vector<Vec2f> lines, Mat &dst) {

	vector<int> centers(dst.rows);

	int midImage = dst.rows/2;
	Point pt1, pt2; double rad_angle;
	Point mid_1(0, midImage);
	Point mid_2(dst.cols, midImage);
	double rad_angle_2;
	line(dst, mid_1, mid_2, red, 1, CV_AA);


	for (size_t i = 0; lines.size() != 0 && i < lines.size() - 1; i++) {
		getInfoLine(lines[i], pt1, pt2, rad_angle);
		Point res;
		bool intersect = intersection(pt1, pt2, mid_1, mid_2, res);
		if (intersect && res.x < dst.rows && res.x > 0 /*&& res.y < dst.cols && res.y > 0*/) {
			centers[res.x] ++;
		}
	}

	int max = 0;
	Point betterCenter;
	betterCenter.y = midImage;
	betterCenter.x = 0;
	for (size_t i = 0; i < centers.size(); i++) {
		int aux = centers[i] /*+ centers[i-1] + centers[i+1]*/;
		if (aux > max) {
			betterCenter.x = i;
			max = aux;
		}
	}

	return betterCenter;
}

/**
 * Calcule vanishing point from lines with indirect vote
 */
Point getCenterIndirect(vector<Vec2f> lines, Mat &dst) {

	vector<Point> centers(lines.size() * lines.size());
	vector<int> numCentersPoint(lines.size() * lines.size());
	int numCenters = 0;
	double error = 5;
	Point pt1, pt2; double rad_angle;
	Point pt1_2, pt2_2; double rad_angle_2;

	for (size_t i = 0; lines.size() != 0 && i < lines.size() - 1; i++) {
		getInfoLine(lines[i], pt1, pt2, rad_angle);

		if (!(std::abs(rad_angle-0) < error || std::abs(rad_angle-180) < error
				 || std::abs(rad_angle-90) < error || std::abs(rad_angle-270) < error)) {  // Vertical or horizontal line

			for (size_t j = i + 1; j < lines.size(); j++) {

				getInfoLine(lines[j], pt1_2, pt2_2, rad_angle_2);

				if (!(std::abs(rad_angle_2-0) < error || std::abs(rad_angle_2-180) < error
												 || std::abs(rad_angle_2-90) < error || std::abs(rad_angle_2-270) < error)) {  // Vertical or horizontal line

					Point res;
					bool intersect = intersection(pt1, pt2, pt1_2, pt2_2, res);

					if (intersect && res.x < dst.rows && res.x > 0 && res.y < dst.cols && res.y > 0) {
						int indexCenter = getClosestCenter(centers, res);
						if (indexCenter != -1) {
							numCentersPoint[indexCenter] ++;
						}
						else{	// New Center
							centers[numCenters] = res;
							numCentersPoint[indexCenter] = 1;
							numCenters ++;
						}

					}

				}

			}	// End for

		}
	}

	int max = 0;
	Point betterCenter;
	for (size_t i = 0; i < numCenters; i++) {
		if (numCentersPoint[i] > max) {
			betterCenter = centers[i];
			max = numCentersPoint[i];
		}
	}

	return betterCenter;
}

int getClosestCenter(vector<Point> centers, Point p) {
	for (size_t i = 0; i < centers.size() - 1; i++) {
		Point diff = centers[i] - p;
		int distance = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
		if (distance < maxDistance) {
			return i;
		}
	}
	return -1;
}

void getInfoLine(Vec2f line, Point &pt1, Point &pt2, double &rad_angle) {
	float rho = line[0], theta = line[1];
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1000 * (-b));
	pt2.y = cvRound(y0 - 1000 * (a));
	rad_angle = (theta * 360) / (2 * CV_PI);
}

void printCross(Mat &src, Point p) {
	int lineLength = 8;
	Point x1 = p; x1.x -= lineLength;
	Point x2 = p; x2.x += lineLength;
	Point y1 = p; y1.y -= lineLength;
	Point y2 = p; y2.y += lineLength;
	line(src, x1, x2, red, 3);
	line(src, y1, y2, red, 3);
}
