// OpenCVtest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "stdio.h"
#include<iostream> 
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<cmath>
using namespace std;
using namespace cv;
const int kThreashold = 220;
const int kMaxVal = 255;
const Size kGaussianBlueSize = Size(5, 5);
Point2f target_centre;
Point2f predictP;
        const int stateNum = 4;					//当前状态值4×1向量(x,y)
        const int measureNum = 4;                               //测量值4×1向量(x,y)	
        KalmanFilter KF(stateNum, measureNum, 0);		//实例化卡尔曼滤波类
        int T = 3;						//卡尔曼滤波参考系下的时间
        Point2f predict_point;					//基于卡尔曼滤波预测的预测点
        int sumT = 0;						//卡尔曼滤波参考系下的时间的总和（用于算法）
        int T_time[3];						//当前帧、上一帧、上上一帧的时间
        int t_count = 0;					//计数
        Point2f llastp, lastp, nowp;				//储存当前帧、上一帧、上上一帧的点信息
        Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

        void kalman_init() {
            KF.transitionMatrix = (Mat_<float>(stateNum, measureNum) <<
                1, 0, T, 0, 
                0, 1, 0, T, 
                0, 0, 1, 0, 
                0, 0, 0, 1
               );  //转移矩阵A

            setIdentity(KF.measurementMatrix);                                  //设置测量矩阵H
            setIdentity(KF.processNoiseCov, Scalar::all(5e-4));                 //设置系统噪声方差矩阵Q
            setIdentity(KF.measurementNoiseCov, Scalar::all(1e-3));             //设置测量噪声方差矩阵R
            setIdentity(KF.errorCovPost, Scalar::all(1));                       //设置后验错误估计协方差矩阵P

        }

        
        Point2f kalman_predict(Point2f target_centre) {  /*此处的target_centre是预测到的装甲板中心*/
            Mat prediction = KF.predict();			                //预测	
            llastp = lastp;					                //在新的一帧里，更新点的信息
            lastp = nowp;
            nowp = target_centre;
            predict_point = Point(prediction.at<float>(0), prediction.at<float>(1));//获取预测值(x',y')
            measurement.at<float>(0) = (float)target_centre.x;			//更新测量值
            measurement.at<float>(1) = (float)target_centre.y;
            measurement.at<float>(2) = (float)nowp.x - lastp.x;                     //速度
            measurement.at<float>(3) = (float)nowp.y - lastp.y;
            //measurement.at<float>(4) = (nowp.x - lastp.x) - (lastp.x - llastp.x);   //加速度
            //measurement.at<float>(5) = (nowp.y - lastp.y) - (lastp.y - llastp.y);
            KF.correct(measurement);			                        //根据测量值修正协方差矩阵
            return predict_point;				                        //返回预测的点
        }
int main()
{
    Mat frame,channels[3],binary,Gaussian;
    Mat cameraMatrix = (Mat_<double>(3, 3) << 1576.70020, 0.000000000000, 635.46084, 0.000000000000, 1575.77707, 529.83878, 0.000000000000, 0.000000000000, 1.000000000000);
    Mat distCoeffs = (Mat_<double>(1, 5) << -0.08325, 0.21277, 0.00022, 0.00033, 0);
    Mat _rVec =Mat::zeros(3, 1, CV_64FC1);//init rvec
	Mat _tVec =Mat::zeros(3, 1, CV_64FC1);//init tvec
    vector<vector<Point>> contours;
    vector<Point2f> point2d;
    vector<Point3f> Points3d;
    vector<Vec4i> hierarchy;    
    Point3f point3f;
    Rect boundRect;
    RotatedRect box;
    double Y_DISTANCE_BETWEEN_GUN_AND_CAM = 0;
    double _xErr, _yErr, _euclideanDistance;
    double BULLET_SPEED = 3000;
    float y_yaw;
	float x_pitch;

    VideoCapture video;
    video.open("/home/ladistra/24-vision-pjh/04/1.jpg装甲板.avi");
    for (;;) {
        Rect point_array[20]; //存放Rect的数组
        video >> frame;
        if (frame.empty()) {
            break;
        }
        /*预处理*/
        split(frame,channels);
        threshold(channels[1], binary, kThreashold, kMaxVal, 0);
        GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
        findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        /*找到灯条*/
        int index = 0;
        for (int i = 0; i < contours.size(); i++) { //遍历轮廓
            //box = minAreaRect(Mat(contours[i]));
            //box.points(boxPts.data());
            boundRect = boundingRect(Mat(contours[i])); //找出轮廓的最小外接正矩形Rect放在boundRect中
            //rectangle(frame, boundRect.tl(), boundRect.br(), (0, 255, 0), 2,8 ,0);
            try
            {       //对boundRect的长宽比，长，宽进行限制
                if (double(boundRect.height / boundRect.width) >= 1.3 && boundRect.height > 36 && boundRect.height>20) {
                    point_array[index] = boundRect; //把找到的灯条Rect放在point_array数组里
                    index++; //记录灯条轮廓
                }
            }
            catch (const char* msg)
            {
                //cout << msg << endl;
                //continue;
            }
        } 
        /*灯条匹配*/
        int point_near[2]; 
        int min = 10000;
        for (int i = 0; i < index-1; i++)
        {
            for (int j = i + 1; j < index; j++) {  //匹配面积大小来匹配灯条
                int value = abs(point_array[i].area() - point_array[j].area());
                if (value < min)
                {
                    min = value;
                    point_near[0] = i;  //把两个灯条对应的轮廓编号记录，用于后续在point_array中找到灯条矩形
                    point_near[1] = j;
                }
            }
        }   
        /*找到装甲板并画线*/
        try
        {   //找到匹配的灯条矩形
            Rect rectangle_1 = point_array[point_near[0]];
            Rect rectangle_2 = point_array[point_near[1]];
            if (rectangle_2.x == 0 || rectangle_1.x == 0) { //剔除不存在的装甲板
                throw "not enough points";
            } //用Rect的长、宽中点找到装甲板角点
            Point point1 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y);
            Point point2 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y + rectangle_1.height);
            Point point3 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y);
            Point point4 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y + rectangle_2.height);
            Point p[4] = { point1,point2,point4,point3 }; 
            cout << p[0]<<p[1]<<p[2]<<p[3] << endl;
            for (int i = 0; i < 4; i++) {
                line(frame, p[i%4], p[(i+1)%4], Scalar(0, 255, 0), 2);
            }      
            line(frame,p[0],p[2],Scalar(255,0,0),2);
            line(frame,p[1],p[3],Scalar(255,0,0),2);

            target_centre=Point2f((point1.x+point4.x)/2,(point1.y+point4.y)/2);
            //circle(frame,target_centre,4,Scalar(0,255,0),-1,8,0);
            kalman_init();
            predictP = kalman_predict(target_centre);
            circle(frame,predictP,4,Scalar(0,255,0),-1,8,0);
            cout << "pT=" << target_centre<<'\t'<<"pP="<< predictP<<endl;
           
            
            //把二维角点存放在向量组里
            point2d.push_back(p[0]); 
            point2d.push_back(p[1]); 
            point2d.push_back(p[2]); 
            point2d.push_back(p[3]); 
        }
        catch (const char* msg)
        {
            //cout << msg << endl;
            //continue;
        }
        //获取三维坐标点
        //自己建立世界坐标系，z一般为0
            // 原点（左下角）
            point3f.x = 0;
            point3f.y = 0;
            point3f.z = 0;
            Points3d.push_back(point3f);
            // 左上角
            point3f.x = 0;
            point3f.y = 5.5;
            point3f.z = 0;
            Points3d.push_back(point3f);
            // 右上角
            point3f.x = 14.0;
            point3f.y = 5.5;
            point3f.z = 0.0;
            Points3d.push_back(point3f);
            // 右下角
            point3f.x = 14;
            point3f.y = 0;
            point3f.z = 0;
            Points3d.push_back(point3f);           

        solvePnP(Points3d, point2d, cameraMatrix, distCoeffs, _rVec, _tVec, false, SOLVEPNP_ITERATIVE);
		_tVec.at<double>(1, 0) -= Y_DISTANCE_BETWEEN_GUN_AND_CAM;
        double x_position=_tVec.at<double>(0, 0);
        double y_position=_tVec.at<double>(1, 0);
        double z_position=_tVec.at<double>(2, 0);
		_euclideanDistance = sqrt(_tVec.at<double>(0, 0)*_tVec.at<double>(0, 0) + _tVec.at<double>(1, 0)*_tVec.at<double>(1, 0) + _tVec.at<double>(2, 0)* _tVec.at<double>(2, 0));
        const auto tan_pitch=_tVec.at<double>(1, 0)/sqrt(_tVec.at<double>(2, 0)*_tVec.at<double>(2, 0)+_tVec.at<double>(0, 0)*_tVec.at<double>(0, 0));
        const auto tan_yaw =_tVec.at<double>(0, 0)/_tVec.at<double>(2, 0);
        x_pitch = -atan(tan_pitch) * 180 / CV_PI;
        y_yaw = atan(tan_yaw) * 180 / CV_PI;
        float camera_target_height = _euclideanDistance * sin(x_pitch / 180 * CV_PI);
        float gun_target_height = camera_target_height + Y_DISTANCE_BETWEEN_GUN_AND_CAM;
        float gun_pitch_tan = gun_target_height / (_euclideanDistance * cos(x_pitch / 180 * CV_PI));
        x_pitch = atan(gun_pitch_tan) / CV_PI * 180;
        string x_pitch0 = to_string(atan(gun_pitch_tan) / CV_PI * 180) ;
        float compensateGravity_pitch_tan = tan(x_pitch/180*CV_PI) + (0.5*9.8*(_euclideanDistance / BULLET_SPEED)*(_euclideanDistance / BULLET_SPEED)) / cos(x_pitch/180*CV_PI);
        x_pitch = atan(compensateGravity_pitch_tan)/CV_PI*180;
        string dis = to_string(_euclideanDistance) ;
        string x_pitch1 = to_string(x_pitch) ;
        putText(frame, dis, point2d[2], 2, 1, Scalar(18, 195, 127));
        //putText(frame,"x_pitch0="+ x_pitch0,point2d[1],2,1,Scalar(18, 195, 127));
        //putText(frame,"x_pitch1="+ x_pitch1,point2d[3],2,1,Scalar(225, 0, 0));

        imshow("video", frame);
        if (waitKey(10) >= 0) {
            break;
        }
    }
    video.release();
   destroyAllWindows();
    return 0;
}