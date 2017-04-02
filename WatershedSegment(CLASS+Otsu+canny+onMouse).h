#ifndef WATERSHEDSEGMENT_H
#define WATERSHEDSEGMENT_H

#include <iostream>  
#include <opencv2\opencv.hpp>  

using namespace cv;  
using namespace std;

Mat srcImage, mouseMasker;  

void onMouse(int event, int x, int y, int flags, void*)
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

class WatershedSegment
{
public:
	WatershedSegment();
	~WatershedSegment() {};
	
	void run(Mat* source);

private:
	Mat srcImageBACKUP;
	Mat binary;
	Mat marker;
	uchar lineMainGray;
	double lineAverageX;
	map<int, Mat> segImg;	//objectLabel,segmentationImage

	
	void init(Mat* source);
	void mouseMark();
	void resetting();
	void processing();
	void calcMouseLine(vector<Point>* _mouseLine);
	uchar findMaxAppear(uchar a[],int n);
	void drawLineOnGuessMarker(vector<Point>* _mouseLine,int label);
	void makeSureFgRow(Point _point,int label);
	void makeSureGuessFg(Point _point,int label);
	void clearControversial();
	void segmentBaseWatershedResult(Mat* watershedResult,int numOfSegment);
	
};

WatershedSegment::WatershedSegment()
{
	lineAverageX = 0;
}

void WatershedSegment::run(Mat* source)
{
	init(source);
	mouseMark();
}

void WatershedSegment::init(Mat* source)
{
	cout<<"Putchar 'r' is resetting!"<<endl;
	cout<<"Putchar 'p' is watershed Processing!"<<endl;

	srcImageBACKUP = source->clone();
	srcImage = srcImageBACKUP.clone(); 
	
	medianBlur(srcImage,srcImage,5);
	cvtColor(srcImage,binary,CV_BGR2GRAY);
	threshold(binary,binary,0,255,THRESH_BINARY + THRESH_OTSU);

	Mat fg,bg;
	erode(binary,fg,Mat(),Point(-1,-1),1);
	//dilate(binary,bg,Mat(),Point(-1,-1),10);
	//threshold(bg,bg,1,128,THRESH_BINARY_INV);
	marker = fg;// + bg;
}

void WatershedSegment::mouseMark()
{
	namedWindow("srcImage",0);
    imshow("srcImage", srcImage);  
	waitKey(50);
    mouseMasker = Mat(srcImage.size(),CV_8UC1,Scalar::all(0));  //mark label, later input findContours()
	setMouseCallback("srcImage", onMouse, 0);  

    while (true)  
    { 
		char chr = waitKey(0);
		if(chr == 27)
		{
			break;
		}
		else if(chr == 'r')
		{
			resetting();
		}
		else if (chr == 'p')
		{
			processing();
			break;
		}
    }  
	destroyAllWindows();
}


void WatershedSegment::processing()
{
	imwrite("srcImage.jpg",srcImage);

	vector<vector<Point>> contoursOfMouseLine;  
	vector<Vec4i> hierarchy;  
	findContours(mouseMasker, contoursOfMouseLine, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);  
	int numOfSegment = contoursOfMouseLine.size();
	if (numOfSegment == 0)  //no contours
	{
		cout<<"No contours!"<<endl;
		return;
	}
	cout << numOfSegment << " contours!" << endl;  
	
	for (int i = 0,bgCnt = 0; i < numOfSegment ; i++)	
	{
		int label = i+1;
		vector<Point> mouseLine = contoursOfMouseLine[i];
		calcMouseLine(&mouseLine);
		
		if(lineMainGray == 0)//background line
		{
			for(int j = 0;j < mouseLine.size();j++)
			{
				marker.at<uchar>(mouseLine[j].y,mouseLine[j].x) = label;
			}
			bgCnt++;
			if(bgCnt > 1)
			{
				cout<<"Too many backgrounds!"<<endl;
				break;
			}
			continue;
		}
		else
		{
			drawLineOnGuessMarker(&mouseLine,label);	//add mouseLine on morphology result
		}
	}
	
	clearControversial();	//clear area: morphology has but mouseLine no (mouseLine is high priority)

	Mat marker32;
	marker.convertTo(marker32,CV_32S);
	watershed(srcImageBACKUP, marker32);  //input type CV_32S
	
	segmentBaseWatershedResult(&marker32,numOfSegment);
}


void WatershedSegment::segmentBaseWatershedResult(Mat* watershedResult,int numOfSegment)
{
	Mat temp;
	// boundaries between the regions is -1, shift right for histogram
	watershedResult->convertTo(temp,CV_8U,1,1);
	
	Mat hist;
	int dims = 1;
	float hrange[] = {0,255};
	const float *ranges[] = {hrange};
	int bins = 256;
	int channels = 0;
	calcHist(&temp,1,&channels,Mat(),hist,dims,&bins,ranges);

	map<int, Mat>::iterator iter;
	for(int i = 0;i < bins;i++)
	{
		if(hist.at<float>(i,0) && i )	//statistics labels and ignore boundaries
		{
			Mat temp = Mat(watershedResult->size(),CV_8U,Scalar::all(0));
			segImg.insert(make_pair(i - 1,temp));
		}
	}
	if(segImg.size() > numOfSegment)
	{
		cout<<"Too many labels!"<<endl;
		return;
	}

	for(int i = 0;i < watershedResult->cols;i++)	//create the masker of objector
	{
		for(int j = 0;j < watershedResult->rows;j++)
		{
			iter = segImg.find(watershedResult->at<int>(j,i));
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
}

void WatershedSegment::drawLineOnGuessMarker(vector<Point>* _mouseLine,int label)
{
	for(int j = 0;j < _mouseLine->size();j++)
	{
		Point pointMouseLine = (*_mouseLine)[j];
		
		if(marker.at<uchar>(pointMouseLine.y,pointMouseLine.x) == label)	//has labeled
		{
			continue;
		}
		else
		{
			if(lineMainGray == marker.at<uchar>(pointMouseLine.y,pointMouseLine.x))
			{
				//confirm foreground
				makeSureFgRow(pointMouseLine,label);
			}else
			{
				//mouseLine say yes, morphology say no
				makeSureGuessFg(pointMouseLine,label);
			}
		}
	}
}


//find the max appear number in array (premise: frequency is greater than 50%)
uchar WatershedSegment::findMaxAppear(uchar a[],int n)  
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


void WatershedSegment::makeSureGuessFg(Point _point,int label)
{
	marker.at<uchar>(_point.y,_point.x) = label;
	if(marker.at<uchar>(_point.y,(int)lineAverageX) == lineMainGray)
	{
		int dist = (int)lineAverageX - _point.x;
		int step = 1;
		if(dist < 0)
		{
			step = -1;
		}
		int tempX = _point.x;
		while (abs(tempX - _point.x) <= abs(dist) || marker.at<uchar>(_point.y,tempX) == lineMainGray)
		{
			marker.at<uchar>(_point.y,tempX) = label;
			tempX += step;
		}
	}
}


void WatershedSegment::makeSureFgRow(Point _point,int label)
{
	int p = 0,q = 0;
	bool left = true,right = true;
	do
	{
		if(left)
		{
			marker.at<uchar>(_point.y,_point.x + p) = label;
			p--;	
			left = marker.at<uchar>(_point.y,_point.x + p) == lineMainGray;
		}
		if(right)
		{
			marker.at<uchar>(_point.y,_point.x + q) = label;
			q++;
			right = marker.at<uchar>(_point.y,_point.x + q) == lineMainGray;
		}
							
	} while (left || right);
}

//calc lineAverageX and lineMainGray
void WatershedSegment::calcMouseLine(vector<Point>* _mouseLine)
{
	uchar* standArray = new uchar[_mouseLine->size()];
	for(int j = 0;j < _mouseLine->size();j++)
	{
		standArray[j] = marker.at<uchar>((*_mouseLine)[j].y,(*_mouseLine)[j].x);
		lineAverageX += (*_mouseLine)[j].x;
	}
	lineAverageX /= _mouseLine->size();
	lineMainGray = findMaxAppear(standArray,_mouseLine->size());
	delete[] standArray;
}

void WatershedSegment::clearControversial()
{
	for(int i = 0;i < marker.cols;i++)
	{
		for(int j = 0;j < marker.rows;j++)
		{
			if( marker.at<uchar>(j,i) == 255)
				marker.at<uchar>(j,i) = 0;
		}
	}
}

void WatershedSegment::resetting()
{
	srcImage = srcImageBACKUP.clone();
	mouseMasker = Scalar::all(0);  
	imshow("srcImage",srcImage);
	waitKey(50);
}

 
#endif