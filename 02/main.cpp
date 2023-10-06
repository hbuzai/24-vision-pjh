// OpenCVtest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
 #include "stdio.h"
#include<iostream> 
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

// 二值化阈值
const int kThreashold = 220; 
const int kMaxVal = 255;
//高斯卷积操作的卷积核尺寸
const Size kGaussianBlueSize = Size(5, 5);

int main()
{
    //VideoCapture video;
    //video.open("/home/ladistra/24-vision-pjh/02/1.png");
    
    //读入图片
    Mat frame = imread("/home/ladistra/24-vision-pjh/02/1.png");

    Mat channels[3],binary,Gaussian,dilatee;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Rect boundRect; //矩形
    RotatedRect box; //旋转矩形
    vector<Point2f> boxPts(4);
    
    /*for (;;) {
        Rect point_array[20];
        video >> frame;
        if (frame.empty()) {
            break;
        }*/
        Rect point_array[20];


        split(frame,channels);
        threshold(channels[0], binary, kThreashold, kMaxVal, 0);
        GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
        dilate(Gaussian, dilatee, element);

        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE); //找到轮廓点

        imshow("pre",dilatee);

        int index = 0;
        for (int i = 0; i < contours.size(); i++) {
            //box = minAreaRect(Mat(contours[i]));
            //box.points(boxPts.data());
            boundRect = boundingRect(Mat(contours[i])); 
            //rectangle(frame, boundRect.tl(), boundRect.br(), (0, 255, 0), 2,8 ,0);
            try
            {
                if (double(boundRect.height / boundRect.width) >= 1.0 && boundRect.height > 36 && boundRect.height>20) {
                    point_array[index] = boundRect;
                    index++;
                }
            }
            catch (const char* msg)
            {
                cout << msg << endl;
                //continue;
            }
        }        
        int point_near[2]; //二维数组保存index索引值
        int min = 10000;
        for (int i = 0; i < index-1; i++)
        {
            for (int j = i + 1; j < index; j++) {
                int value = abs(point_array[i].area() - point_array[j].area());
                if (value < min)
                {
                    min = value;
                    point_near[0] = i;
                    point_near[1] = j;
                }
            }
        }   
        try
        {
            Rect rectangle_1 = point_array[point_near[0]];
            Rect rectangle_2 = point_array[point_near[1]];
            if (rectangle_2.x == 0 || rectangle_1.x == 0) {
                throw "not enough points";
            }
            Point point1 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y);
            Point point2 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y + rectangle_1.height);
            Point point3 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y);
            Point point4 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y + rectangle_2.height);
            Point p[4] = { point1,point2,point4,point3 };
            cout << p[0]<<p[1]<<p[2]<<p[3] << endl;
            for (int i = 0; i < 4; i++) {
                line(frame, p[i%4], p[(i+1)%4], Scalar(0, 255, 0), 2);
            }      
            line(frame,p[0],p[2],Scalar(255,0,0),1);
            line(frame,p[1],p[3],Scalar(255,0,0),1);      
        }
        catch (const char* msg)
        {
            cout << msg << endl;
            //continue;
        }

        /*imshow("video", frame);
        if (waitKey(10) >= 0) {
            break;
        }
    }
    video.release();
    cv::destroyAllWindows();*/
    imshow("out",frame);
    waitKey(0);
    return 0;
}


/*
#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
//由于在识别中的核心物体以及相关的物理特性是灯条，所以建一个灯条类
class LightDescriptor
{	    //在识别以及匹配到灯条的功能中需要用到旋转矩形的长宽偏转角面积中心点坐标等
public:float width, length, angle, area;
      cv::Point2f center;
public:
    LightDescriptor() {};
    //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
    LightDescriptor(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};
 
int main()
{
    //VideoCapture video; //VC类对象化
    //video.open("/home/geek_style/桌面/ood_red.mp4");
    Mat frame = imread("/home/ladistra/24-vision-pjh/02/1.png");

    //变量集中定义
    Mat channels[3], binary, Gaussian, dilatee;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Rect boundRect;
    RotatedRect box;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point2f> boxPts(4);
    //图像预处理
    //for (;;) {
        //Rect point_array[20];
        //video >> frame;  //读取每帧
        //if (frame.empty()) {
           // break;
        //}
        split(frame, channels); //通道分离
        threshold(channels[0], binary, 220, 255, 0);//二值化
        GaussianBlur(binary, Gaussian, Size(5, 5), 0);//滤波
        dilate(Gaussian, dilatee, element);
        // dilate(Gaussian, dilate, element, Point(-1, -1));//膨胀，把滤波得到的细灯条变宽
        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
        vector<LightDescriptor> lightInfos;//创建一个灯条类的动态数组
    //筛选灯条
        for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积
            double area = contourArea(contours[i]);
            // 去除较小轮廓&fitEllipse的限制条件
            if (area < 5 || contours[i].size() <= 1)
                continue;//相当于就是把这段轮廓去除掉
            // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
            RotatedRect Light_Rec = fitEllipse(contours[i]);
 
            // 长宽比和轮廓面积比限制（由于要考虑灯条的远近都被识别到，所以只需要看比例即可）
            if (Light_Rec.size.width / Light_Rec.size.height > 4)
                continue;
            lightInfos.push_back(LightDescriptor(Light_Rec));
        }
        //二重循环多条件匹配灯条
        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                LightDescriptor& leftLight = lightInfos[i];
                LightDescriptor& rightLight = lightInfos[j];
                float angleGap_ = abs(leftLight.angle - rightLight.angle);
                //由于灯条长度会因为远近而受到影响，所以按照比值去匹配灯条
                float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
                //均长
                float meanLen = (leftLight.length + rightLight.length) / 2;
                float lengap_ratio = abs(leftLight.length - rightLight.length) / meanLen;
                float yGap = abs(leftLight.center.y - rightLight.center.y);
                float yGap_ratio = yGap / meanLen;
                float xGap = abs(leftLight.center.x - rightLight.center.x);
                float xGap_ratio = xGap / meanLen;
                float ratio = dis / meanLen;
                //匹配不通过的条件
                if (angleGap_ > 15 ||
                    LenGap_ratio > 1.0 ||
                    lengap_ratio > 0.8 ||
                    yGap_ratio > 1.5 ||
                    xGap_ratio > 2.2 ||
                    xGap_ratio < 0.8 ||
                    ratio > 3 ||
                    ratio < 0.8) {
                    continue;
                }
                //绘制矩形
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
                Point2f vertices[4];
                rect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0,255,0), 2.2);
                }
            }
        }
 
        namedWindow("video", WINDOW_FREERATIO);
        imshow("video", frame);
        waitKey(0);
        return 0;
    }
    //video.release();
    //cv::destroyAllWindows();*/
    