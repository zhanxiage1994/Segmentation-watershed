#include<iostream>
#include<opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("haha.jpg");
	if(img.empty())
		return 0;
	medianBlur(img,img,5);

	Mat binary;
	cvtColor(img,binary,CV_BGR2GRAY);
	threshold(binary,binary,30,255,THRESH_BINARY);// + THRESH_OTSU);
	
	Mat fg,bg;
	erode(binary,fg,Mat(),Point(-1,-1),1);
	dilate(binary,bg,Mat(),Point(-1,-1),1);
	threshold(bg,bg,1,128,THRESH_BINARY_INV);
	Mat marker = fg + bg;

	Mat canny;
	Canny(marker,canny,110,150);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	findContours(canny,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);//CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE
	imwrite("marker.jpg",marker);
	Mat marker32;
	marker.convertTo(marker32,CV_32S);
	watershed(img,marker32);
	convertScaleAbs(marker32,marker32);
	Mat thresh,threshInv;
	threshold(marker32,thresh,0,255,THRESH_BINARY + THRESH_OTSU);
	bitwise_not(thresh,threshInv);
	Mat res,res3;
	bitwise_and(img,img,res,thresh);
	bitwise_and(img,img,res3,threshInv);
	Mat res4;
	addWeighted(res,1,res3,1,0,res4);
	drawContours(res4,contours,-1,Scalar(0,255,0),1);
	namedWindow( "Watershed", 0);
    imshow( "Watershed", res4 );
    waitKey(0);
	imwrite("Watershed.jpg",res4);

	//Mat binary;
	//cvtColor(image1,binary,CV_BGR2GRAY);
	//threshold(binary,binary,30,255,THRESH_BINARY);
	////namedWindow("binary",0);
	////imshow("binary",binary);
	////waitKey();

	//Mat element5(5,5,CV_8U,Scalar(1));
	//Mat fg1;
	//erode(binary,fg1,Mat(),Point(-1,-1),6);
	////morphologyEx(binary,fg1,MORPH_OPEN,element5);
	////morphologyEx(fg1,fg1,MORPH_CLOSE,element5);
	////namedWindow("Foreground",0);
	////imshow("Foreground",fg1);
	////waitKey();

	//Mat bg1;
	//dilate(fg1,bg1,Mat(),Point(-1,-1),6);
	//threshold(bg1,bg1,1,128,THRESH_BINARY_INV);
	////namedWindow("Background",0);
	////imshow("Background",bg1);
	////waitKey();

	//Mat markers1 = fg1 + bg1;
	////namedWindow("markers",0);
	////imshow("markers",markers1);
	////waitKey();

	//WatershedSegment segmenter1;
	//segmenter1.setMarkers(markers1);
	//segmenter1.process(image1);
	////namedWindow("Segmentation");
	////imshow("Segmentation",segmenter1.getSegmentation());
	////waitKey();
	//Mat maskImage = segmenter1.getSegmentation();
	//threshold(maskImage,maskImage,250,1,THRESH_BINARY);
	//cvtColor(maskImage,maskImage,COLOR_GRAY2BGR);
	//maskImage = image1.mul(maskImage);


	system("pause");
	return 0;
}