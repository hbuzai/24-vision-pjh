#include "stdio.h"
#include<iostream> 
#include<vector>
//#include "include.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;
#define FRONT_BACK_PARA    2.0f 
/**
     * 大符速度特点 加速/减速
     */
    enum SPEED_TYPE {
        SPEED_UP = 0,
        SPEED_DOWN
    };  
/**
     * 大符转速结构体
     */
    struct RotateSpeed {
        float lastRotateSpeed = 0.0f;
        float nowRotateSpeed = 0.0f;
        float realRotateSpeed = 0.0f;
        SPEED_TYPE speedType;
    };
        struct SpeedRange {
        float nowMinSpeed = 100.0f;
        float realMinSpeed = 0.0f;
        float nowMaxSpeed = 0.0f;
        float realMaxSpeed = 0.0f;
        int minSameNumber = 0;
        int maxSameNumber = 0;
        bool minSpeedFlag = false;
        bool maxSpeedFlag = false;
    };
    struct ParaCircle {
    float buffPredictAngle = 25.0f;
    float buffRadBias = 2.0f;
    float realCantileverLength = 730.0f;
    float armorAngleBias = 5.0f;
    float minToArmorDistance = 220;//80;
    float maxToArmorDistance = 600;
    float minArea = 80.0f;
    float maxArea = 400;
    float minLengthWidthRatio = 0.7f;
    float maxLengthWidthRatio = 1.5f;
};
    ParaCircle paraCircle;
    static ParaCircle getParaCircle() {
        return paraCircle;
    }
     /**
     * 神符旋转方向
     */
    enum DIR_OF_ROTA {
        STOP = 0,
        CLOCKWISE,
        ANTICLOCKWISE,
        UNKNOW,
    };
       /**
     * 大符速度特点 加速/减速
     */
    enum SPEED_TYPE {
        SPEED_UP = 0,
        SPEED_DOWN
    };
       /**
     * 大符转速结构体
     */
    struct RotateSpeed {
        float lastRotateSpeed = 0.0f;
        float nowRotateSpeed = 0.0f;
        float realRotateSpeed = 0.0f;
        SPEED_TYPE speedType;
    };

    bool _captureFlag = false;
    bool _isBig = false;//大符标志位
    bool _sameTargetFlag = false;//是否切换叶片标志位
    //BuffSolverStruct _buffSolverStruct;
    //ArmorRealData _buffArmor = ArmorDataFactory::getBuffArmor();
    RotateSpeed _rotateSpeed;
    SpeedRange _speedRange;
    //SineFunction _sineFunction;
    ParaCircle _paraCircle = BuffParaFactory::getParaCircle();
    float _circleAngle180 = 0.0f;
    float _circleAngle360 = 0.0f;
    float _realAddAngle = 0.0f;
    float _para = 0.0f;
    float _armorRatio = 1.78f;
    float _delayTime = 0.4f;
    bool _resetBuff = false;//重置预判标志位
    bool _buffInit = false;//是否初始化成功标志位
    vector<float> _buffAngleList;
    DIR_OF_ROTA _buffDirOfRota = UNKNOW;
    cv::Point2f _predictCoordinate;
    cv::Point2f _predictBias = cv::Point2f(0.0f, 0.0f);
    //BuffBias _buffBias = NO_CHANGE;

  

    /**
     * 计算点pt1,点pt2,与点pt0所形成的夹角的角度
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
    /**
     * 圆周率
     */
    static inline double PI() {
        return CV_PI;
    }
        /**
     * 圆周率
     */
    static inline float PI_F() {
        return float(CV_PI);
    }
        /**
     * 角度转弧度
     * @param p 角度值
     * @return 弧度值
     */
    static double angleToRadian(double p) {
        return p * PI() / 180.0f;
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
    //Mat hsvImage,HsvImage;
    vector<Mat> channels;
    Point2f p2;
    Point2f target;/*目标点*/

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
                            //Rect minRect = boundingRect(contours[i]);
                            //circle(frame,minRect.center,15,Scalar(5,255,255));
                            //circle(frame,minrect.center,15,Scalar(5,255,255));
                            //cout << minrect.center<<endl;
                            p2 = minrect.center;
                            }
                    } 
                    else {
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
            //cout << p2 << endl;
            //cout << R << endl;
            //circle(frame,p2,R*1.123,Scalar(255,0,255),1,8,0);
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

            //预测开始
            if (_rotateSpeed.realRotateSpeed <= 0.0f) {
        return float();
    }
    //spd=0.785*sin(1.884*t)+1.305
    float possibleTime[2];
    float realTime;
    //速度超限要改
    if (_rotateSpeed.realRotateSpeed < _sineFunction.para - _sineFunction.amplitude) {
        _rotateSpeed.realRotateSpeed = _sineFunction.para - _sineFunction.amplitude;
    }
    if (_rotateSpeed.realRotateSpeed > _sineFunction.para + _sineFunction.amplitude) {
        _rotateSpeed.realRotateSpeed = _sineFunction.para + _sineFunction.amplitude;
    }
    possibleTime[0] = (asinf((_rotateSpeed.realRotateSpeed - _sineFunction.para) / _sineFunction.amplitude)) / _sineFunction.rotateIndex;
    possibleTime[1] = (possibleTime[0] > 0 ? Util::PI_F() / (_sineFunction.rotateIndex) - possibleTime[0] : Util::PI_F() / (-_sineFunction.rotateIndex) - possibleTime[0]);
   // cout << "    " << _rotateSpeed.speedType << endl;
    realTime = (_rotateSpeed.speedType == SPEED_UP ? possibleTime[0] : possibleTime[1]);
    //cout << "time--" << fabs(possibleTime[0] - possibleTime[1]) << endl;
    
            //初始数据处理
            static float lastCircleAngle = 0.0f;
            static Point2f lastTargetRectCenter = Point2f(0.0f, 0.0f);
            static float circleAngleBias = 0.0f;
            _circleAngle180 = getAngle(target, Point2f(3000.0f, p2.y), p2);
                //旋转角度处理
            if (p2.y < target.y) {
                _circleAngle360 = 360.0f - _circleAngle180;
                _circleAngle180 = -_circleAngle180;
            } else {
                _circleAngle360 = _circleAngle180;
            }
                //神符方向重置
            if (_resetBuff) {
                if (_buffDirOfRota != UNKNOW) {
                    _buffDirOfRota = UNKNOW;
                    _buffInit = false;
                    _buffAngleList.resize(0);
                    _resetBuff = false;
                }
            }
            circleAngleBias = _circleAngle360 - lastCircleAngle;
                //判断是否是同一片叶片
            if ((fabsf(circleAngleBias) > 10.0f || (fabsf(lastTargetRectCenter.y - target.y) > 40.0f) || (fabsf(lastTargetRectCenter.x - target.x) > 40.0f))) {
                _sameTargetFlag = false;
            }
            else {
                _sameTargetFlag = true;
            }
            if (!_buffInit) {
                //过去角度处理
                if (!_sameTargetFlag) {
                    _buffAngleList.resize(0);//角度变化过大认为叶片跳变
                } else {
                    _buffAngleList.push_back(_circleAngle360);
                }
                //数据足够判断方向

                if (_buffAngleList.size() > 5) {//存五个数据用来判断
                    float averageAngle = 0.0f;
                    float buff = 0.0f;
                    uint8_t count = 0;
                    //逐差法计算
                    for (size_t i = 0; i < _buffAngleList.size() / 2; i++) {
                        buff = _buffAngleList[_buffAngleList.size() / 2 + i] - _buffAngleList[i];
                        if (!buff) {
                            continue;
                        } else {
                            averageAngle += buff;
                            count++;
                        }
                    }
                    if (count) {
                        averageAngle = averageAngle / count;
                    }
                    //判断旋转方向
                    if (averageAngle < -1.0f) {
                        _buffDirOfRota = CLOCKWISE;
                    } else if (averageAngle > 1.0f) {
                        _buffDirOfRota = ANTICLOCKWISE;
                    } else {
                        _buffDirOfRota = STOP;
                    }
                    _buffInit = true;
                }
            }
           // cout<<"yeah"<<_predictCoordinate<<endl;
            lastCircleAngle = _circleAngle360;
            lastTargetRectCenter = target;
            //转速
            //定义静态过去和现在角度；
            static double nowAngle = 0.0f;
            static double lastAngle = 0.0f;
            static int count = 0;
            //定义过去和现在时间
            static double lastTime = (double) cv::getTickCount() / cv::getTickFrequency() * 1000; // ms
            double curTime = (double) cv::getTickCount() / cv::getTickFrequency() * 1000;
            //如果叶片没有跳变，则把过去和现在角度以及过去和现在速度置零
            lastAngle = nowAngle = _rotateSpeed.lastRotateSpeed = _rotateSpeed.nowRotateSpeed = 0.0f;
            //如果过去角度已经被清零，则过去角度进行初始化为现在绝对角度
            if (lastAngle == 0.0f) {
                lastAngle = _circleAngle360;
            }
            //每0.1s一次数据刷新
            if (curTime - lastTime < 100) {    
            }
            //帧数递增
            count++;
            nowAngle = _circleAngle360;
            //计算实时角速度
            _rotateSpeed.nowRotateSpeed = (float) fabs(angleToRadian((nowAngle - lastAngle)) * (1000.0f / (curTime - lastTime)));
            //过去角度和时间更新
            lastAngle = nowAngle;
            lastTime = curTime;
            //如果过去角速度已被清零，则对过去速度进行更新
            if (_rotateSpeed.lastRotateSpeed == 0.0f) {
                _rotateSpeed.lastRotateSpeed = _rotateSpeed.nowRotateSpeed;
                
            }
            //防止出现异常数据
            if (_rotateSpeed.nowRotateSpeed > 5 || _rotateSpeed.nowRotateSpeed < -5) {
                
            }
            //如果速度没有替换最小速度，则计数加1
            if (_speedRange.nowMinSpeed > _rotateSpeed.nowRotateSpeed) {
                _speedRange.nowMinSpeed = _rotateSpeed.nowRotateSpeed;
            } else {
                _speedRange.minSameNumber++;
            }
            //如果速度没有替换最大速度，则计数加1
            if (_speedRange.nowMaxSpeed < _rotateSpeed.nowRotateSpeed) {
                _speedRange.nowMaxSpeed = _rotateSpeed.nowRotateSpeed;
            } else {
                _speedRange.maxSameNumber++;
            }
            //如果连续20帧没有刷新最小速度，则该速度为波谷速度（该速度一旦更新，便不再更新）
            if (_speedRange.minSameNumber > 20 && !_speedRange.minSpeedFlag) {
                _speedRange.realMinSpeed = _speedRange.nowMinSpeed;
                _speedRange.minSpeedFlag = true;
            }
            //如果连续20帧没有刷新最大速度，则该速度为波峰速度（该速度一旦更新，便不再更新）
            if (_speedRange.maxSameNumber > 20 && !_speedRange.maxSpeedFlag) {
                _speedRange.realMaxSpeed = _speedRange.nowMaxSpeed;
                _speedRange.maxSpeedFlag = true;
            }
            //赋值真实速度，方便后面使用
            _rotateSpeed.realRotateSpeed = _rotateSpeed.nowRotateSpeed;
            _rotateSpeed.speedType = (_rotateSpeed.nowRotateSpeed > _rotateSpeed.lastRotateSpeed ? SPEED_UP : SPEED_DOWN);}

            //转角
            float time = 0.0f;
            //计算实时角速度
            //calculateRotateSpeed();
            time= realTime;
            //小符
            _paraCircle.buffPredictAngle += FRONT_BACK_PARA;

            //pre
            float r = 0.0f;
            float addAngle = 0.0f;
            float resAngle = 0.0f;
            static cv::Point2f predictBias = cv::Point2f(0.0f, 0.0f);
            addAngle =_paraCircle.buffPredictAngle;
            if (_buffDirOfRota) {
            resAngle = -_circleAngle180 + addAngle;
            if (_circleAngle180 >= 180.0f) {
                resAngle = resAngle - 360.0f;
            } else if (resAngle < -180.0f) {
                resAngle = 360.0f + resAngle;
            }
            //位置预判
            r = distance(p2, target)*1.08;
            _predictCoordinate.x = p2.x + r * cosf(resAngle * PI_F() / 180.0f);
            _predictCoordinate.y = p2.y + r * sinf(resAngle * PI_F() / 180.0f);
            cout<<"1="<< _predictCoordinate <<endl;
            }
            
            vector<Point2f> cirV;
            //在得到装甲板中心点后将其放入缓存队列中
            //用于拟合圆，用30个点拟合圆
            if(cirV.size()<30)
            {    cirV.push_back(p2);}
            else{  
            cirV.erase(cirV.begin());}
            //将打击点围绕圆心旋转某一角度得到预测的打击点
            if(p2.x!=0&&p2.y!=0){
                for(int j=1;j<5;j++){
                    //得到旋转一定角度(这里是30度)后点的位置    
                Mat rot_mat=getRotationMatrix2D(p2,72*j,1);
                float sinA=rot_mat.at<double>(0,1);
                //sin(30);    
                float cosA=rot_mat.at<double>(0,0);
                //cos(30);    
                float xx=-(p2.x-target.x);
                float yy=-(p2.y-target.y);
                Point2f resPoint=Point2f(p2.x+cosA*xx-sinA*yy,p2.y+sinA*xx+cosA*yy); 
                circle(frame, resPoint, 3, Scalar(0, 255, 0), -1, 8, 0);
                line(frame,resPoint,p2,Scalar(0,255,0),1);
              
                }  
                      
            }


            //imshow("pre1", pre1);
            //imshow("pre", pre);
               imshow("video", frame);
            if (waitKey(10) >= 0) {
                break;
            }   
            

        
        video.release();
        destroyAllWindows();
        return 0;


    }
    

 
    

    
    

    


