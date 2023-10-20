#include "stdio.h"
#include<iostream> 
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
using namespace std;
using namespace cv;

/*** 计算点pt1,点pt2,与点pt0所形成的夹角的角度
     * @param pt1 点1
     * @param pt2 点2
     * @param pt0 点0
     * @return 夹角角度
     */
    static float getAngle(cv::Point2f pt1, cv::Point2f pt2, cv::Point2f pt0) {
        float dx1 = pt1.x - pt0.x;
        float dy1 = pt1.y - pt0.y;
        float dx2 = pt2.x - pt0.x;
        float dy2 = pt2.y - pt0.y;
        float angle_line = (dx1 * dx2 + dy1 * dy2) / sqrtf((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10f);
        return acosf(angle_line) * 180.0f / 3.141592653f;
    }
/** * 圆周率
    */
    static float PI_F() {
        return float(CV_PI);
    }

/*** 圆周率
    */
    static double PI() {
        return CV_PI;
    }
/*** 角度转弧度
    * @param p 角度值
    * @return 弧度值
    */
    static double angleToRadian(double p) {
        return p * PI() / 180.0f;
    }
/*** 弧度转角度
   * @param p 弧度值
   * @return 角度值
   */
    static inline float radianToAngle(float p) {
        return p * 180.0f / PI_F();
    }
void drawRotatedRect(Mat &frame, const RotatedRect &rect, const Scalar &color, int thickness)
{
    Point2f Vertex[4];
    rect.points(Vertex);
    for(int i = 0 ; i < 4 ; i++)
    {
        line(frame, Vertex[i] , Vertex[(i + 1) % 4] , color , thickness);
    }
}
double distance(Point2f a,Point2f b)
{
    return sqrt((a.x -b.x)*(a.x -b.x) + (a.y -b.y)*(a.y -b.y));
}

int main(){

    Mat frame,bin,pre,pre1;
    vector<Mat> channels;
    Point2f p2;

    VideoCapture video;
    video.open("/home/ladistra/24-vision-pjh/123/avi/rune-detect.avi");

    for(;;){
        Point2f point_array[10];

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

                if(width / height < 5) {
                    contours_min_rects.push_back(minrect);
                    if(minrect.size.area() > 100 && minrect.size.area() < 500 && minrect.center.y > 500 &&minrect.center.y < 520 && minrect.center.x > 680&&minrect.center.x <700 ){ // find R
                        if(height / width > 0.85){
                            r_min_rects.push_back(minrect);
                            p2 = minrect.center;
                            }
                    } 
                }   
            }
        }
        //再次处理图片以找到未激活扇叶的轮廓
        Mat element = getStructuringElement(MORPH_RECT,Size(9,9));
        dilate(pre1,pre,element); //膨胀使轮廓连接起来

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
        }
            double R;
            Point2f p1 =Point2f(rectMid);      
            R = distance(p1,p2);
            //cout << p2 << endl;
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
            circle(frame, target, 4, Scalar(0, 255, 0), -1, 8, 0);
            //circle(frame, p2, 3, Scalar(0, 255, 255), -1, 8, 0);
            line(frame,target,p2,Scalar(0,255,0),1);

            //预测开始
                float resAngle = 0.0f;
                float _circleAngle360=0.0f;
                vector<float> _buffAngleList;
                //float r = 0.0f;
                cv::Point2f _predictCoordinate;
                //static float lastCircleAngle = 0.0f;
                //static cv::Point2f lastTargetRectCenter = cv::Point2f(0.0f, 0.0f);
                //static float circleAngleBias = 0.0f;
                static double addAngle=0.0f;
                //circleAngleBias = _circleAngle360 - lastCircleAngle;
                //定义静态过去和现在角度；
                static double nowAngle = 0.0f;
                static double lastAngle = 0.0f;
                static double deltaAngle;
                //static double lastRotateSpeed = 0.0f;
                static double nowRotateSpeed = 0.0f;
                //定义过去和现在时间；
                static double lastTime = (double) cv::getTickCount() / cv::getTickFrequency() * 1000; // ms
                double curTime = (double) cv::getTickCount()*3/ cv::getTickFrequency() * 1000;
                //cout << lastTime <<endl;
                float _circleAngle180 = getAngle(target, cv::Point2f(3000.0f, p2.y), p2);
                //cout << "1=" <<_circleAngle180<<endl;
                //旋转角度处理，显示离设定的轴正向的夹角，即转角；
                if (p2.y < target.y) {
                    _circleAngle360 = 360.0f - _circleAngle180;
                    _circleAngle180 = -_circleAngle180;
                } else {
                    _circleAngle360 = _circleAngle180;
                }
                _buffAngleList.push_back(_circleAngle360);
                //cout << "2=" <<_circleAngle360<<endl;
                    for(int i=0;i<_buffAngleList.size();i++){
                        lastAngle = _buffAngleList[i];
                        nowAngle = _buffAngleList[i+3];
                        //计算实时角速度
                        nowRotateSpeed = (float) fabs(angleToRadian((nowAngle - lastAngle)) * (1000.0f / (curTime - lastTime)));
                        resAngle = 1000*radianToAngle(nowRotateSpeed);//一帧1s；
                        //cout << "3="<<resAngle<<endl;
                        Mat rot_mat2=getRotationMatrix2D(target,resAngle,1);
                        float sinB=rot_mat2.at<double>(0,1);    
                        float cosB=rot_mat2.at<double>(0,0);   
                        float xx=-(p2.x-target.x);
                        float yy=-(p2.y-target.y);
                        Point2f _predictCoordinate=Point2f(p2.x+cosB*xx-sinB*yy,p2.y+sinB*xx+cosB*yy); 
                        circle(frame, _predictCoordinate, 4, Scalar(255, 255, 255), -1, 8, 0);
                        //cout<<"9=" << _predictCoordinate <<endl;
                        line(frame,_predictCoordinate,p2,Scalar(255,255,255),1); 
                    }
            //将打击点围绕圆心旋转某一角度得到预测的打击点
            if(p2.x!=0&&p2.y!=0){
                for(int j=1;j<5;j++){  
                Mat rot_mat=getRotationMatrix2D(p2,72*j,1);
                float sinA=rot_mat.at<double>(0,1);  
                float cosA=rot_mat.at<double>(0,0);    
                float xx=-(p2.x-target.x)*0.95;
                float yy=-(p2.y-target.y);
                Point2f resPoint=Point2f(p2.x+cosA*xx-sinA*yy,p2.y+sinA*xx+cosA*yy); 
                circle(frame, resPoint, 3, Scalar(255,0, 0), -1, 8, 0);
                line(frame,resPoint,p2,Scalar(255,0,0),1); 
                }                        
            }
            //imshow("pre1", pre1);
            //imshow("pre", pre);
            imshow("video", frame);
            if (waitKey(10) >= 0) {
                break;
            }
            }               
    video.release();
    destroyAllWindows();
    return 0;

    }
    

    


