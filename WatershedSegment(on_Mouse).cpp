#include <iostream>  
#include <opencv2\opencv.hpp>  
  
using namespace std;  
using namespace cv;  
  
Mat srcImage, maskImage;  


  
void on_Mouse(int event, int x, int y, int flags, void*)  
{  
	static Point clickPoint;   
     
    if (x < 0 || x >= srcImage.cols || y < 0 || y >= srcImage.rows)  
        return;  
  

    if (event == EVENT_LBUTTONDOWN)  
    {  
        clickPoint = Point(x, y);  
    }         
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))  
    {  
        Point point(x, y);  
        line(maskImage, clickPoint, point, Scalar::all(255), 5, 8, 0);  
        line(srcImage, clickPoint, point, Scalar::all(255), 5, 8, 0);  
        clickPoint = point;  
        imshow("srcImage", srcImage);  
    }  
}  
 
int main()  
{  
	cout<<"Putchar 'r' is resetting!"<<endl;
	cout<<"Putchar 'p' is watershed Processing!"<<endl;

    srcImage = imread("haha.jpg");  
    Mat srcImageBACKUP = srcImage.clone();  
    maskImage = Mat(srcImage.size(),CV_8UC1,Scalar::all(0));  

	namedWindow("srcImage",0);
    imshow("srcImage", srcImage);  
	waitKey(50);
    setMouseCallback("srcImage", on_Mouse, 0);  

	Mat maskWaterShed;
    while (true)  
    {  
		char chr = waitKey(0);
		if (chr == 'p')
		{
			vector<vector<Point>> contours;  
			vector<Vec4i> hierarchy;  
			findContours(maskImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);  
			if (contours.size() == 0)  
				break;  
			cout << contours.size() << " contours!" << endl;  

			maskWaterShed = Mat(maskImage.size(), CV_32S, Scalar::all(0));  
			
			for (int index = 0; index < contours.size(); index++) 
			{
				drawContours(maskWaterShed, contours, index, Scalar::all(index + 1), -1, 8, hierarchy, INT_MAX);  
			}
			watershed(srcImageBACKUP, maskWaterShed);  
  
			vector<Vec3b> colorTab;  
			for (int i = 0; i < contours.size(); i++)  
			{  
				int b = theRNG().uniform(0, 255);  
				int g = theRNG().uniform(0, 255);  
				int r = theRNG().uniform(0, 255);  
  
				colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));  
			}  
  
			Mat resImage = Mat(srcImage.size(), CV_8UC3);
			for (int i = 0; i < maskImage.rows; i++)  
			{ 
				for (int j = 0; j < maskImage.cols; j++) 
				{   
					int index = maskWaterShed.at<int>(i, j);   
					if (index == -1)  	//boundaries 
						resImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);  
					else if (index <= 0 || index > contours.size())  
						resImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);  
					else 
						resImage.at<Vec3b>(i, j) = colorTab[index - 1]; 
				}  
			}  
			namedWindow("resImage",0);
			imshow("resImage", resImage);  
			waitKey(30);
			imwrite("resImage.jpg",resImage);
			addWeighted(resImage, 0.3, srcImageBACKUP, 0.7, 0, resImage); 
			namedWindow("WaterShed",0);
			imshow("WaterShed", resImage);  
			waitKey(30);
			break;
		}
		else if(chr == 27)
		{
			break;
		}
    }  
	system("pause");
    return 0;  
}  
  
