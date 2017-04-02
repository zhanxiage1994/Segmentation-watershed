#include <iostream>  
#include <opencv2\opencv.hpp>  
  
using namespace std;  
using namespace cv;  
  
Mat srcImage, mouseMasker;  

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
        line(mouseMasker, clickPoint, point, Scalar::all(255), 5, 8, 0);  
        line(srcImage, clickPoint, point, Scalar::all(255), 5, 8, 0);  
        clickPoint = point;  
        imshow("srcImage", srcImage);  
    }  
}  

//find the max appear number in array (premise: frequency is greater than 50%)
uchar FindIt(uchar a[],int n)  
{  
	uchar ch;  
	int times=0;  
	for(int i = 0;i < n;i++)  
	{  
		if(times == 0)
		{  
			ch = a[i];  
			times = 1;  
		}  
		else 
		{
			if(ch == a[i])
			{
				times++;  
			}
			else
			{
				times--;  
			}
		}
	}  
	return ch;  
}  

int main()  
{  
	cout<<"Putchar 'r' is resetting!"<<endl;
	cout<<"Putchar 'p' is watershed Processing!"<<endl;

    srcImage = imread("haha.jpg");  
    Mat srcImageBACKUP = srcImage.clone(); 
	
	medianBlur(srcImage,srcImage,5);
	Mat binary;
	cvtColor(srcImage,binary,CV_BGR2GRAY);
	threshold(binary,binary,0,255,THRESH_BINARY + THRESH_OTSU);

	Mat fg,bg,marker;
	erode(binary,fg,Mat(),Point(-1,-1),1);
	//dilate(binary,bg,Mat(),Point(-1,-1),10);
	//threshold(bg,bg,1,128,THRESH_BINARY_INV);
	//Mat marker = fg + bg;
	fg.copyTo(marker);

	namedWindow("srcImage",0);
    imshow("srcImage", srcImage);  
	waitKey(50);
    mouseMasker = Mat(srcImage.size(),CV_8UC1,Scalar::all(0)); 	//mark label, later input findContours()
	setMouseCallback("srcImage", on_Mouse, 0);  

    while (true)  
    { 
		char chr = waitKey(0);
		if (chr == 'p')
		{
			imwrite("srcImage.jpg",srcImage);
			vector<vector<Point>> contours;  
			vector<Vec4i> hierarchy;  
			findContours(mouseMasker, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);  
			if (contours.size() == 0)  // no contours
				break;  
			cout << contours.size() << " contours!" << endl;  
			Mat haha = mouseMasker;
			for (int i = 0,bgCnt = 0; i < contours.size() ; i++)	
			{
				uchar* standArray = new uchar[contours[i].size()];
				double averageX = 0;
				for(int j = 0;j < contours[i].size();j++)
				{
					standArray[j] = marker.at<uchar>(contours[i][j].y,contours[i][j].x);
					averageX += contours[i][j].x;
				}
				averageX /= contours[i].size();
				uchar standard = FindIt(standArray,contours[i].size());
				
				if(standard == 0)//background line
				{
					for(int j = 0;j < contours[i].size();j++)
					{
						marker.at<uchar>(contours[i][j].y,contours[i][j].x) = (i+1);
					}
					bgCnt++;
					delete[] standArray;
					if(bgCnt > 1)
					{
						cout<<"Too many backgrounds!"<<endl;
						return -1;
					}
					continue;
				}
				else
				{
					for(int j = 0;j < contours[i].size();j++)
					{
						if(marker.at<uchar>(contours[i][j].y,contours[i][j].x) == (i+1))	//has labeled
						{
							continue;
						}
						if(standard == standArray[j])	//confirm foreground
						{
							int p = 0,q = 0;
							bool left = true,right = true;
							do
							{
								if(left)
								{
									marker.at<uchar>(contours[i][j].y,contours[i][j].x + p) = (i+1);
									p--;	
									left = marker.at<uchar>(contours[i][j].y,contours[i][j].x + p) == standard;
								}
								if(right)
								{
									marker.at<uchar>(contours[i][j].y,contours[i][j].x + q) = (i+1);
									q++;
									right = marker.at<uchar>(contours[i][j].y,contours[i][j].x + q) == standard;
								}
							
							} while (left || right);
						}else	//mouseLine say yes, morphology say no
						{
							marker.at<uchar>(contours[i][j].y,contours[i][j].x) = (i+1);
							if(marker.at<uchar>(contours[i][j].y,averageX) == standard)
							{
								int dist = averageX - contours[i][j].x;
								int step = 1;
								if(dist < 0)
								{
									step = -1;
								}
								int tempX = contours[i][j].x;
								while (abs(tempX - contours[i][j].x) <= abs(dist) || marker.at<uchar>(contours[i][j].y,tempX) == standard)
								{
									marker.at<uchar>(contours[i][j].y,tempX) = (i+1);
									tempX += step;
								}
							}
						}
					}
					delete[] standArray;
				}
				//drawContours(maskWaterShed, contours, index, Scalar::all(index + 1), -1, 8, hierarchy, INT_MAX);  

			}
			for(int i = 0;i < marker.cols;i++)
			{
				for(int j = 0;j < marker.rows;j++)
				{
					if( marker.at<uchar>(j,i) == 255)
						marker.at<uchar>(j,i) = 0;
				}
			}
			imwrite("marker.jpg",marker);
			//Mat marker = imread("marker.jpg",0);
			Mat marker32;
			marker.convertTo(marker32,CV_32S);
			watershed(srcImageBACKUP, marker32);  //input type CV_32S
			Mat temp;
			// boundaries between the regions is -1, shift right for histogram
			marker32.convertTo(temp,CV_8U,1,1);/
			Mat hist;
			int dims = 1;
			float hrange[] = {0,255};
			const float *ranges[] = {hrange};
			int bins = 256;
			int channels = 0;
			calcHist(&temp,1,&channels,Mat(),hist,dims,&bins,ranges);

			map<int ,Mat> segImg;
			map<int,Mat>::iterator iter;
			for(int i = 0;i < bins;i++)
			{
				if(hist.at<float>(i,0) && i )	//ignore boundaries
				{
					Mat temp = Mat(marker32.size(),CV_8U,Scalar::all(0));
					segImg.insert(make_pair(i - 1,temp));
				}
			}
			if(segImg.size() > contours.size())
			{
				cout<<"Too many labels!"<<endl;
				return -1;
			}
			for(int i = 0;i < marker32.cols;i++)
			{
				for(int j = 0;j < marker32.rows;j++)
				{
					iter = segImg.find(marker32.at<int>(j,i));
					if(iter != segImg.end())
					{
						iter->second.at<uchar>(j,i) = 1;
					}
				}
			}
			for(iter = segImg.begin();iter != segImg.end();iter++)
			{
				temp = Scalar::all(0);
				char buf[20];
				sprintf_s(buf,"%d.jpg",iter->first);
				bitwise_and(srcImageBACKUP,srcImageBACKUP,temp,iter->second);
				imwrite(buf,temp);
			}
			break;
		}
		else if(chr == 27)
		{
			break;
		}
		else if(chr == 'r')
		{
			srcImage = srcImageBACKUP.clone();
			mouseMasker = Scalar::all(0);  
			imshow("srcImage",srcImage);
			waitKey(50);
		}
    }  
	destroyAllWindows();
	system("pause");
    return 0;  
}  
  
