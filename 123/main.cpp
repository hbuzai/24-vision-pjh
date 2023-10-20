#include "stdio.h"
#include<iostream> 
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

void drawRotatedRect(Mat &frame, const RotatedRect &rect, const Scalar &color, int thickness)
{
    Point2f Vertex[4];
    rect.points(Vertex);
    for(int i = 0 ; i < 4 ; i++)
    {
        line(frame , Vertex[i] , Vertex[(i + 1) % 4] , color , thickness);
    }
}
double distance(Point2f a,Point2f b)
{
    return sqrt((a.x -b.x)*(a.x -b.x) + (a.y -b.y)*(a.y -b.y));
}
int main(){

    Mat frame,bin,pre,pre1;
    //Mat hsvImage,HsvImage;
    vector<Mat> channels;
    Point2f p2;

    VideoCapture video;
    video.open("/home/ladistra/24-vision-pjh/123/avi/rune-detect.avi");

    for(;;){

        video >> frame;
        if (frame.empty()){
            break;
        }

        //预处理
        split(frame,channels);     
        Mat sframe = channels.at(2) - channels.at(0); 
        threshold(sframe,bin,100,255,THRESH_BINARY);
        GaussianBlur(bin,pre1,Size(5,5),0);
        //查找轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchies;
        findContours(pre1, contours, hierarchies, RETR_TREE, CHAIN_APPROX_NONE);
        //drawContours(frame, contours, -1, Scalar(0, 255, 0), 1, 8);

        //找到圆周运动的圆心——R
        int minArea = 10000;
        int minId;
        Point2f center;  /*定义外接圆中心坐标*/
        float radius;  /*定义外接圆半径*/  

        vector<RotatedRect> contours_min_rects;//所有轮廓的最小外接矩形
        vector<RotatedRect> armor_min_rects;
        vector<RotatedRect> target_min_rects;
        vector<RotatedRect> r_min_rects; //R
                   
        //遍历轮廓
        for (int i = 0; i < contours.size(); i++) {
            vector<Point>points;
            double area = contourArea(contours[i]);/*面积排除噪声*/
            if (area < 10 || area>10000)
                continue;
            if (hierarchies[i][3] >= 0 && hierarchies[i][3] < contours.size()) /*找到没有父轮廓的轮廓*/
                continue;
		    if (hierarchies[i][2] >= 0 && hierarchies[i][2] < contours.size())/*找没子轮廓的轮廓*/
			    continue;
            
            RotatedRect minrect = minAreaRect(contours[i]);
            Rect rect = boundingRect(contours[i]); 
            //宽和高匹配
            if (minrect.size.area() <= 6000.0 && minrect.size.area()>150) {
                float width;
                float height;
                if (minrect.size.width > minrect.size.height){
                    width = minrect.size.width;
                    height = minrect.size.height;
                } else {
                    width = minrect.size.height;
                    height = minrect.size.width;
                }

            if (width / height < 5) {
                contours_min_rects.push_back(minrect);
                if(minrect.size.area() > 100 && minrect.size.area() < 500 && minrect.center.y > 500 &&minrect.center.y < 520 && minrect.center.x > 680&&minrect.center.x <700 ){ // find R
                    if(height / width > 0.85){
                        r_min_rects.push_back(minrect);
                        //Rect minRect = boundingRect(contours[i]);
                        //circle(frame,minRect.center,15,Scalar(5,255,255));
                        //circle(frame,minrect.center,15,Scalar(5,255,255));
                        //cout << minrect.center<<endl;
                        p2 = minrect.center;
                        }
                     } else {
                        if(minrect.size.area() > 300 && minrect.size.area() < 4350 && (height / width) < 0.7) {
                            armor_min_rects.push_back(minrect);
                        }     
                    }
                }   
            }
        }

       /*找到轮廓*/
        Mat element = getStructuringElement(MORPH_RECT,Size(9,9));
        dilate(pre1,pre,element);

        vector<vector<Point>> contours2;
        vector<Vec4i> hierarchies2;
        findContours(pre, contours2, hierarchies2, RETR_TREE, CHAIN_APPROX_NONE);
        //drawContours(frame, contours2, -1, Scalar(0, 255, 0), 1, 8);
	    double maxArea = -1;
	    int maxId;
        Point2f rectMid;
       
        for (int i = 0; i < contours2.size(); i++) {
            double area = contourArea(contours2[i]);
		if (area < 5000 || area>10000)/*面积排除噪声*/
			continue;
		if (hierarchies2[i][3] >= 0 && hierarchies2[i][3] < contours2.size())/*找到没有父轮廓的轮廓*/
			continue;
		if (hierarchies2[i][2] >= 0 && hierarchies2[i][2] <= 10)/*找子轮廓多于10的*/
			continue;
        int num = hierarchies2[i][2];
        int max_index = -1;
        if (max_index <= num){
            max_index = num;
            maxId = i;
            }
        }   

        /*半径参考长度所在轮廓几何中心*/
            if (maxId >= 0 && maxId < contours2.size()) {
                /*画出需打部位轮廓*/
                //drawContours(frame, contours2, maxId, Scalar(0, 255, 0), 1, 8);
                //RotatedRect target = minAreaRect(contours2[maxId]);
                //drawRotatedRect(frame,target,Scalar(255,0,0),1);
                Moments rect;
                rect = moments(contours2[maxId], false);            /*计算矩*/
                Point2f rectmid;
                rectmid = Point2f(rect.m10 / rect.m00, rect.m01 / rect.m00);            /*计算中心矩:*/
                rectMid = rectmid;
                //circle(frame,rectMid, 1, Scalar(0, 255, 255), -1, 8, 0);
                }   

        double R;
        Point2f p1 =Point2f(rectMid);      
        R = distance(p1,p2);
        //cout << R << endl;
        //circle(frame,p2,R*1.123,Scalar(255,0,255),1,8,0);

        Point2f target;/*目标点*/
	    double multiple = 1.123;/*倍率，换算目标点所用*/
       
	
		/*第一象限*/
	if (rectMid.x >= p2.x && rectMid.y <= p2.y) {
		target = Point2f(p2.x + (rectMid.x - p2.x) * multiple, p2.y - (p2.y - rectMid.y) * multiple);
 
	}
	/*第二象限*/
	if (rectMid.x <= p2.x && rectMid.y <= p2.y) {
		target = Point2f(p2.x - (p2.x - rectMid.x) * multiple, p2.y - (p2.y - rectMid.y) * multiple);
 
	}
	/*第三象限*/
	if (rectMid.x <= p2.x && rectMid.y >= p2.y) {
		target = Point2f(p2.x - (p2.x - rectMid.x) * multiple, p2.y + (rectMid.y - p2.y) * multiple);
 
	}
	/*第四象限*/
	if (rectMid.x >= p2.x && rectMid.y >= p2.y) {
		target = Point2f(p2.x + (rectMid.x - p2.x) * multiple, p2.y + (rectMid.y - p2.y) * multiple);
 
	}
	circle(frame, target, 3, Scalar(255, 255, 0), -1, 8, 0);
	circle(frame, p2, 3, Scalar(0, 255, 255), -1, 8, 0);
    line(frame,target,p2,Scalar(255,0,0),1);


        imshow("pre1", pre1);
        imshow("pre", pre);
        imshow("video", frame);
        if (waitKey(10) >= 0) {
            break;
        }

    }
    video.release();
    destroyAllWindows();
    return 0;
    

    }


