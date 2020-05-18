#include <iostream>
#include <fstream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

using namespace cv;
using namespace std;

int main ()
{
	Mat img;
	img=imread("images/00000014.png");
	namedWindow("Fisheye");
	imshow("Fisheye", img);
	waitKey(0);
	return 0;
}
