#include <stdio.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void cameraCalibration();
Mat panorama(Mat img1, Mat img2);
Mat getImage(VideoCapture capture);
Mat removeBlack(Mat img);
Mat padding(Mat src);

bool camera = true;
bool calibrate = false;
bool useCalibration = false;
std::string calibrate_parameters = "calibrate_camera.yml";
double rescaleFactor = 0.6;
int numRescale = 1;

Mat intrinsic;
Mat distCoeffs;


int main(int argc, char** argv) {

	if (calibrate) {
		cameraCalibration();
	}
	else {
		if (!camera) {

			vector<Mat> images;

			//images.push_back(imread("data/panorama_image1.jpg"));
			//images.push_back(imread("data/panorama_image2.jpg"));


			images.push_back(imread("data/g.jpg"));
			images.push_back(imread("data/f.jpg"));
			images.push_back(imread("data/e.jpg"));


			/*
			images.push_back(imread("data/pan4.png"));
			images.push_back(imread("data/pan3.png"));
			images.push_back(imread("data/pan2.png"));
			images.push_back(imread("data/pan1.png"));
			*/

			Mat result = images[0];

			for (int i=1; i< images.size(); i++) {
				//imshow("0", result);
				//imshow("1", images[i]);
				result = panorama(result, images[i]);
				result = removeBlack(result);
				Mat aux = result.clone();
				resize(aux, aux, cvSize(0, 0), 0.5/numRescale, 0.5/numRescale);
				imshow("Result", aux);
				numRescale += 0.5;

				waitKey();
			}

		}
		else {
			printf("Camera\n");

			FileStorage fs = FileStorage(std::string("calibrate_camera.yml"),
						FileStorage::READ);
			fs["intrinsic"] >> intrinsic;
			fs["distCoeffs"] >> distCoeffs;
			fs.release();

			char key = 0;

			VideoCapture capture;
			capture.open(0);
			if (!capture.isOpened()) {
				capture.open(0);
				if (!capture.isOpened()) {
					return -1;
				}
			}

			waitKey(2000);

			printf("Camerar2\n");
			Mat img1 = getImage(capture);
			printf("Photo\n");
			img1 = getImage(capture);

			waitKey(30);

			while (key != 27) {
				//waitKey(500);
				waitKey(0);
				Mat img2 = getImage(capture);

				img1 = panorama(img1, img2);
				img1 = removeBlack(img1);
				//imshow("Panorama", img1);

				Mat aux = img1.clone();
				resize(aux, aux, cvSize(0, 0), 0.5/numRescale, 0.5/numRescale);
				imshow("Panorama", aux);
				numRescale += 0.5;

				key = cv::waitKey(20);
			}

			waitKey();
		}
	}

}

Mat removeBlack(Mat img) {

	int right = 0;
	// Remove right zone
	bool good = true;
	for (int i=img.cols-1; i>=0 && good; i--){
		for (int j=0; j<img.rows && good; j++) {
			if (img.at<Vec3b>(j,i)[0] != 0 || img.at<Vec3b>(j,i)[1] != 0 || img.at<Vec3b>(j,i)[2] != 0) {
				good = false;
			}
		}
		if (good) {
			right ++;
		}
	}

	int high = 0;
	good = true;
	for (int i=img.rows-1; i>=0 && good; i--){
		for (int j=0; j<img.cols && good; j++) {
			if (img.at<Vec3b>(i,j)[0] != 0 || img.at<Vec3b>(i,j)[1] != 0 || img.at<Vec3b>(i,j)[2] != 0) {
				good = false;
				printf("AAAAA %d %d \n", i,j);
			}
		}
		if (good) {
			high ++;
		}
	}

	printf("high: %d\n", high);

	return img(Rect(0, 0, img.cols-right, img.rows-high));
}


void cameraCalibration() {

	int numBoards = 25;
	int numCornersHor = 8;
	int numCornersVer = 5;

	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);
	VideoCapture capture = VideoCapture(0);

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > image_points;

	vector<Point2f> corners;
	int successes=0;

	Mat image;
	Mat gray_image;
	capture >> image;

	vector<Point3f> obj;
	for(int j=0;j<numSquares;j++)
	    obj.push_back(Point3d(j/numCornersHor, j%numCornersHor, 0.0f));

	while (image.empty()) capture >> image;

	while(successes<numBoards)
	{

	    cvtColor(image, gray_image, CV_BGR2GRAY);

	    bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

	    if(found)
	    {
	        cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	        drawChessboardCorners(gray_image, board_sz, corners, found);

	        image_points.push_back(corners);
			object_points.push_back(obj);
			printf("Snap stored!\n");

			successes++;

			waitKey(200);

			if(successes>=numBoards)
				break;
	    }

	    imshow("win1", image);
	    imshow("win2", gray_image);

	    capture >> image;

	    int key = waitKey(1);

	    if(key==27)
	        return;

	}

	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;

	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;

	printf("Calibrating\n");
	calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
	printf("Calibrated\n");

	Mat imageUndistorted;
	FileStorage fs = FileStorage(calibrate_parameters, FileStorage::WRITE);
	fs << "intrinsic" << intrinsic;
	fs << "distCoeffs" << distCoeffs;
	fs.release();

	while(1)
	{
	    capture >> image;
	    undistort(image, imageUndistorted, intrinsic, distCoeffs);

	    imshow("win1", image);
	    imshow("win2", imageUndistorted);

	    waitKey(1);
	}

	capture.release();

	return;

}

//https://ramsrigoutham.com/2012/11/22/panorama-image-stitching-in-opencv/
Mat panorama(Mat color1, Mat color2) {

	printf("=======================================================\n");
	Mat img1, img2, aux1, aux2;
	cvtColor(color1, aux1, CV_BGR2GRAY);
	cvtColor(color2, aux2, CV_BGR2GRAY);
	if (camera && useCalibration) {
	    undistort(aux1, img1, intrinsic, distCoeffs);
	    undistort(aux2, img2, intrinsic, distCoeffs);
	}
	else {
		img1 = aux1.clone();
		img2 = aux2.clone();
	}

	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	//vector<DMatch > matches;
	vector<vector<DMatch > > matches;

	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);
	//matcher.match(descriptors1, descriptors2, matches);

	/*FlannBasedMatcher matcher;
	vector<DMatch > matches;
	matcher.match( descriptors1, descriptors2, matches );*/

	double max_dist = 0;
	double min_dist = 100;

	for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i][0].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
		if (dist > max_dist) {
			max_dist = dist;
		}
	}

	/*for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
		if (dist > max_dist) {
			max_dist = dist;
		}
	}*/

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	std::vector<DMatch> good_matches;
	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i][0].distance < 0.8*matches[i][1].distance
				&& matches[i][0].distance <= max(2 * min_dist, 0.02)) {
			good_matches.push_back(matches[i][0]);
		}
	}

	/*for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance <= max(2 * min_dist, 0.02)) {
			good_matches.push_back(matches[i]);
		}
	}*/


	int kp = 0;
	for (int i = 0; i < good_matches.size(); i++) {
		if (keypoints1[good_matches[i].queryIdx].pt.x > keypoints2[good_matches[i].trainIdx].pt.x) {
			kp ++;
		}
	}
	printf("%d %d\n", kp, good_matches.size());

	bool changed = false;
	if (kp > good_matches.size()*0.5) {
		printf("Mirroring\n");
		Mat m_aux = img1.clone();
		swap(img1, img2);
		swap(color1, color2);

		vector<KeyPoint> kp_aux = keypoints1;
		keypoints1 = keypoints2;
		keypoints2 = kp_aux;

		m_aux = descriptors1;
		descriptors1 = descriptors2;
		descriptors2 = m_aux;

		int i_aux;
		for (int i = 0; i < good_matches.size(); i++) {
			i_aux = good_matches[i].queryIdx;
			good_matches[i].queryIdx = good_matches[i].trainIdx;
			good_matches[i].trainIdx = i_aux;
		}

		changed = true;
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++) {
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);
	resize(img_matches, img_matches, cvSize(0, 0), 0.5/numRescale, 0.5/numRescale);
	imshow("matches", img_matches);
	waitKey(0);

	printf("Homography %d %d \n", obj.size(), scene.size());

	if (obj.size() >= 4) {
			Mat H = findHomography(obj, scene, CV_RANSAC);
			Mat result;
			warpPerspective(color1, result, H, Size((img1.cols + img2.cols)*2, (img1.rows + img1.rows)*2));
			Mat half(result, cv::Rect(0, 0, img2.cols, img2.rows));
			color2.copyTo(half);
			return result;
	}
	else {
		printf("NULL\n");
		if (changed) return color2;
		else return color1;
	}
}

Mat getImage(VideoCapture capture) {
	Mat image;
	while (capture.grab() && image.empty() && capture.retrieve(image));
	return image;
}

Mat padding(Mat src) {
	cv::Mat padded;
	padded.create(src.rows*2, src.cols*2, src.type());
	padded.setTo(cv::Scalar::all(0));

	src.copyTo(padded(Rect(src.cols/2, src.rows/2, src.cols, src.rows)));
	return padded;
}
