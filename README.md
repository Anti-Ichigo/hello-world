
我要将我大学写下的代码保存下来：
#include<iostream>
using namespace std;

double arctan(double x)
{
	double sqr = x*x;
	double e = x;
	double r = 0;
	int i = 1;
	while (e / i > 1e-15)
	{
		double f = e / i;
		r = (i % 4 == 1) ? r + f : r - f;
		e = e*sqr;
		i += 2;
	}
	return r;
}

int main()
{
	double a = 16.0*arctan(1/ 5.0);
	double b = 4.0*arctan(1/ 239.0);
	cout << "PI=" << a - b << endl;
	return 0;
}


//时钟类
#include<iostream>
using namespace std;

	class Clock//时钟类的定义
	{
	public:
		void setTime(int newH = 0, int newM = 0, int newS = 0);
		void showTime();
	private:
		int hour, minute, second;
	};

	//时钟类成员的具体实现
	void Clock::setTime(int newH,int newM,int newS)
	{
		hour = newH;
		minute = newM;
		second = newS;
	}

	inline void Clock::showTime()
	{
		cout << hour << ":" << minute << ":" << second << endl;
	}
	int main()
{
		Clock myclock;//定义对象myclock；
		cout << "First time set and output:" << endl;
		myclock.setTime();//设置时间为默认值
		myclock.showTime();//显示时间
		cout << "Second time set and output:" << endl;
		myclock.setTime(8, 30, 30);
		myclock.showTime();

	system("pause");
	return 0;
}


//hanoi塔问题
#include<iostream>
using namespace std;

/*
(1)将A上n-1个盘子移到B上（借助C）
（2）把A上剩下的一个盘子移到C盘上
（3）把n-1个盘子从B移到C（借助A）
*/;
//move1表示将将src上最上面的一个盘子移到dst上
void move1(char src,char dst)
{
	cout << src << "==>" << dst << endl;
}
//movemore表示将n个盘子从src针上移到dst针上(借助medium)
void movemore(int n, char src, char medium, char dst)
{
	if (n == 1)
		move1(src, dst);
	else
	{
		movemore(n - 1, src, dst, medium);//(1)
		move1(src, dst);//(2)
		movemore(n-1,medium, src, dst);//(3)
	}
}
int main()
{
	int n;
	cout << "input n" << endl;
	cin >> n;
	movemore(n, 'A', 'B', 'C');

	system("pause");
	return 0;
}


//产生随机数
#include<iostream>
#include<time.h>
using namespace std;
int fun1(int x)
{
	return x*x;
}
int fun2(int x,int y)
{
	return fun1(x) + fun1(y);
}
int main()
{
	int a, b;
	cout << "input" << endl;
	cin >> a >> b;
	cout << a << "  " << b<<endl;
	cout << fun2(a, b) << endl;
	system("pause");
	return 0;
}


//从m个数中选k个数
#include<iostream>
using namespace std;

int fun(int m,int k)
{
	int y;
	if ((k == 0) || (m == k))
	{
		return 1;
	}
	else
		y = fun(m - 1, k) + fun(m - 1, k - 1);
	return y;
}
int main()
{
	int m, k;
	cin >> m >> k;
	cout << endl<<fun(m, k);
	

	system("pause");
	return 0;
}
#include<iostream>
using namespace std;

int fun(int n)
{
	int y;
	if (n == 0)
		y= 1;
	else if (n>0)
	{
		y = n*fun(n - 1);
	}
	return y;
}
int main()
{
	int x;
	cin >> x;
	cout << fun(x);
	system("pause");
	return 0;
}
#include<iostream>
using namespace std;

double arctan(double x)
{
	double s = x;
	int i = 1;
	double r = 0;
	while (s / i > 1e-15)
	{
		double f = s / i;
		
		r=(i % 4 == 1 )? r + f : r - f;
		s = s*x*x;
		i += 2;
	}
	return r;
}

int main()
{
	double a = 16.0*arctan(1/ 5.0);
	double b = 4.0*arctan(1/ 239.0);
	cout << "PI=" << a - b << endl;
	return 0;
}

//寻找11~999的回文数
#include<iostream>
#define N 20
using namespace std;

int fun(int x)
{
	int m = 0,n,y;
	y = x;//保留x的初始值
	while (x > 0)
	{
		n = x % 10;
		x = x / 10;
		m = m * 10 + n;
	}
	return (m ==y);
}

int main()
{
	int i;
	for (i = 11; i <= 999; i++)
	{
		if ((fun(i)) && (fun(i*i)) && (fun(i*i*i)))
		{
			cout << "  x=" << i;
			cout << "  x*x=" << i*i;
			cout << "  x*x*x=" << i*i*i << endl;
		}
	}

	return 0;
}

//寻找11~999的回文数
#include<iostream>
#include<cmath>
using namespace std;

int main()
{
	int r, s;
	double k;
	cout << "input r,s" << endl;
	cin >> r >> s;
	if (r*r <= s*s)
		k = sqrt(sin(r)*sin(r) + sin(s)*sin(s));
	else
	{
		k = 1.0 / 2 * sin(r*s);

	}
	cout<<endl << k;
	return 0;
}

//产生随机数
#include<iostream>
#include<time.h>
using namespace std;

int main()
{
	int number;
	srand((unsigned)time(NULL));
	for (int i = 0; i < 20; i++)
	{
		number = rand() % 6;
		cout << number << endl;
	}
	return 0;
}


#include<stdlib.h>
#include<opencv2\opencv.hpp>
#include<opencv2\gpu\gpu.hpp>

int main()
{
	int num_devices = cv::gpu::getCudaEnabledDeviceCount();
	if (num_devices <= 0)
	{
		std::cerr << "There is no devoce" << std::endl;
		return -1;
	}
	int enable_device_id = -1;
	for (int i = 0; i < num_devices; i++)
	{
		cv::gpu::DeviceInfo dev_info(i);
		if (dev_info.isCompatible())
		{
			enable_device_id = i;
		}
	}
	if (enable_device_id < 0)
	{
		std::cerr << "GPU module isn't built for GPU" << std::endl;
		return -1;
	}

	cv::gpu::setDevice(enable_device_id);

	cv::Mat src_image = cv::imread("yuan.jpg");
	if (!src_image.data)
	{
		printf("error\n");
		exit(1);
	}
	cv::Mat dst_image;
	cv::gpu::GpuMat d_src_img(src_image);//upload src image to gpu  
	cv::gpu::GpuMat d_dst_img;
	printf("good luck");
	cv::gpu::cvtColor(d_src_img, d_dst_img, CV_BGR2GRAY);//////////////////canny  
	d_dst_img.download(dst_image);//download dst image to cpu  
	cv::imshow("yuan", dst_image);
	cv::waitKey(50000);

	return 0;
}
//使用指针遍历图像
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat grayim(600, 800, CV_8UC1);
	Mat colorim(600, 800, CV_8UC3);

	//遍历单通道图像像素
	for (int i = 0; i < grayim.rows; i++)
	{
		//获取第i行首像素的地址
		uchar *p = grayim.ptr<uchar>(i);
		//对第i行的每个像素操作
		for (int j = 0; j < grayim.cols; j++)
			p[j] = (i + j) % 255;
	}

	//遍历3通道彩色图像
	for (int i = 0; i < colorim.rows; i++)
	{
		Vec3b *p = colorim.ptr<Vec3b>(i);
		for (int j = 0; j < colorim.cols; j++)
		{
			p[j][0] = i % 255;
			p[j][1] = j % 255;
			p[j][2] = 0;
		}
	}

	
	imshow("colorim", colorim);
	imshow("grayim", grayim);
	waitKey(20000);

	return 0;
}		
						*****定义ROI区域******
#include<opencv2\opencv.hpp>

using namespace cv;

int main()
{
	Mat src = imread("lena.jpg");
	Mat roi = src(Rect(200, 180,200, 150));

	imshow("lena", roi);
	waitKey(5000);
	return 0;
}

						*****矩阵元算*****
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat A = Mat::eye(2, 2, CV_64FC1);//创建一个4行4列的单位矩阵;
	Mat B = A * 3 + 1;
	Mat C = B.diag(0) + B.col(0);//提取B的主对角线diag（0）+B的第1列
	Mat D =(A)*(B);

	cout << "A=" << A << endl << endl;
	cout << "B=" << B << endl << endl;
	cout << "C=" << C << endl << endl;
	cout << "A.*B=" << A.dot(B) << endl;
	cout << "D=" << D << endl << endl;
	return 0;
}

					****使用低通滤波器****
//使用低通滤波器
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("car.jpg");
	Mat dst;
	Mat dstG;
	cv::blur(src, dst, cv::Size(5, 5));
	cv::GaussianBlur(src, dstG, cv::Size(5, 5),1.5);
	
	imshow("car", src);
	imshow("car2", dst);
	imshow("carG", dstG);
	waitKey(50000);

	return 0;
}
//改变图像大小
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("car.jpg");
	Mat dst;
	Mat dstG;
	Mat dst2;
	
	cv::pyrDown(src, dst);//pyrDown将图像的尺寸减半
	cv::pyrUp(src, dstG);//pyrUp将图像尺寸放大一倍
	//cv::resize可任意改变图像大小
	cv::resize(src, dst2, cv::Size(src.rows / 3, src.cols / 3));//改变为1/3的大小

	imshow("car", src);
	imshow("cart", dst);
	imshow("carG", dstG);
	imshow("car2", dst2);
	waitKey(50000);

	return 0;
}

//椒盐噪点
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void salt(Mat &image, int n)//椒盐噪点
{
	for (int k = 0; k < n; k++)
	{
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		if (image.channels() == 1)//灰度图
		{
			image.at<uchar>(j, i) = 255;
		}
		else if (image.channels() == 3)//彩色图
		{
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}

}

int main()
{
	Mat src = imread("car.jpg");
	Mat dst = imread("car.jpg");
	//调用函数增加噪点
	salt(dst, 3000);

	imwrite("salt.jpg", dst);
	imshow("car2",dst);
	imshow("car", src);
	waitKey(20000);

	waitKey(50000);

	return 0;
}

//中值滤波器medianBlur
#include<opencv2\opencv.hpp>
#include<iostream>
using namespace cv;

int main()
{
	Mat src = imread("salt.jpg");//读取椒盐噪点的图像
	Mat dst;

	cv::medianBlur(src, dst, 5);

	imshow("car2",dst);
	imshow("car", src);
	waitKey(20000);

	return 0;
}

//使用Sobel滤波器
#include<opencv2\opencv.hpp>
#include<iostream>
#include<opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("car.jpg");
	Mat sobelX;
	Mat sobelY;
	Mat sobel;
	cv::Sobel(src,sobelX, CV_8U, 1, 0, 3, 0.4, 128);
	cv::Sobel(src, sobelY, CV_8U, 0, 1, 3, 0.4, 128);
	//合并结果
	sobel = abs(sobelX) + abs(sobelY);
	//搜索Sobel极大值
	double sobmin, sobmax;
	cv::minMaxLoc(sobel, &sobmin, &sobmax);
	//变换为8位图像
	Mat sobelImage;
	sobel.convertTo(sobelImage, CV_8U, -255. / sobmax, 255);
	//阈值化处理，得到二值图像
	Mat sobelThresholded;
	cv::threshold(sobelImage, sobelThresholded,70, 255, cv::THRESH_BINARY);

	imshow("car", sobelThresholded);

	waitKey(50000);

	return 0;
}


//应用Canny算法
#include<opencv2\opencv.hpp>

using namespace cv;

int main()
{
	Mat src = imread("road.jpg");
	Mat dst;
	Mat dst2;
	cv::Canny(src, dst, 125, 350);//后面的数字分别表示低阈值和高阈值
	//反转黑白值
	cv::threshold(dst, dst2, 128, 255, cv::THRESH_BINARY_INV);

	imshow("road", src);
	imshow("road2", dst);
	imshow("road3", dst2);
	waitKey(20000);

	return 0;
}

#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int trackbar = 0;
int trackbar2 = 0;
int pos;
// 回调函数
void on_trackbar(int , void*)
{
	Mat src = imread("road.jpg");
	Mat dst;
	Canny(src, dst, trackbar,trackbar2);
	imshow("效果图窗口", dst);
}
int main()
{
	Mat img = imread("road.jpg", 0);
	
	//创建滑动条之前, 要先创建一个窗体，以便把创建的滑动条放置在上面:
	namedWindow("效果图窗口",1);
	
	createTrackbar("低阈值", "效果图窗口", &trackbar, 500, on_trackbar);
	createTrackbar("高阈值", "效果图窗口", &trackbar, 500, on_trackbar);
	on_trackbar(trackbar, 0);//轨迹回调函数
	on_trackbar(trackbar2, 0);

	waitKey(0);
	
	return 0;
}


#include<opencv2\opencv.hpp>
#include <iostream> 
using namespace std;
using namespace cv;
int trackbar=80; //对比度值  
int trackbar2=0;  //亮度值  

// 回调函数
void on_trackbar(int, void*)
{
	Mat src = imread("001.jpg");
	Mat dst;
	dst = Mat::zeros(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
	{
		for (int x= 0; x < src.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{//saturate_cast<uchar>为强制类型转换
				dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((trackbar*0.01)*(src.at<Vec3b>(y, x)[c]) + trackbar2);
			}
		}
	}
	imshow("原图", src);
	imshow("效果图窗口", dst);
}
int main()
{
	namedWindow("效果图窗口", 1);
	createTrackbar("对比度", "效果图窗口", &trackbar, 300, on_trackbar);
	createTrackbar("亮度", "效果图窗口", &trackbar2, 200, on_trackbar);
	on_trackbar(trackbar, 0);
	on_trackbar(trackbar2, 0);

	waitKey(0);
	return 0;
}


     
    //使用霍夫变换检测直线  (书中代码，下一个代码是最近改动的)
    #include<opencv2\opencv.hpp>  
    #include <iostream>   
    using namespace std;  
    using namespace cv;  
      
    int main()  
    {  
        Mat src = imread("road.jpg");  
        if (!src.data){ printf("error"); exit(0); };  
        Mat dst, med;  
        dst = Mat::zeros(src.size(), src.type());  
        //应用Canny算法  
        Canny(src, med, 125, 350);  
      
        cvtColor(med, dst, CV_GRAY2BGR);//将med转换为8位单通道2进制图像dst  
        //Hough变换检测直线  
        std::vector<Vec2f>lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
        HoughLines(med, lines, 1, CV_PI / 180, 80);  
          
        //使用迭代器进行图像遍历  
        std::vector<cv::Vec2f>::const_iterator begin = lines.begin();  
        while (begin != lines.end())  
        {  
            float rho = (*begin)[0];//第一个参数为距离ρ  
            float theta = (*begin)[1];//第二个参数为角度θ  
            if (theta<CV_PI / 4. || theta>3.*CV_PI / 4.)//垂直线  
            {  
                //线与第1行的交点  
                cv::Point pt1(rho / cos(theta), 0);  
                //线与最后一行的交点  
                cv::Point pt2((rho - dst.rows*sin(theta)) / cos(theta), dst.rows);  
                //绘制白线  
                cv::line(dst, pt1, pt2, cv::Scalar(255,255,255), 1);  
            }  
            else//水平线  
            {  
                //线与第1列的交点  
                cv::Point pt1(0, rho / sin(theta));  
                //线与最后一列的交点  
                cv::Point pt2(dst.cols, (rho - dst.cols*cos(theta)) / sin(theta));  
                //绘制白线  
                cv::line(dst, pt1, pt2, cv::Scalar(255,255,255), 1);  
            }  
            begin++;  
        }  
      
        imshow("result", dst);  
        //imshow("med", med);  
        waitKey(0);  
      
        return 0;  
    }  

    //使用霍夫变换检测直线  
    #include<opencv2\opencv.hpp>  
    #include <iostream>   
    using namespace std;  
    using namespace cv;  
      
    int main()  
    {  
        Mat src = imread("road.jpg");  
        if (!src.data){ printf("error"); exit(0); };  
        Mat dst, med;  
        dst = Mat::zeros(src.size(), src.type());  
        //应用Canny算法  
        Canny(src, med, 125, 350);  
      
        cvtColor(med, dst, CV_GRAY2BGR);//<span style="color:#FF6666;">将med转换为8位单通道2进制图像dst</span>  
        //Hough变换检测直线，源图像需为8位单通道2进制图像  
        std::vector<Vec2f>lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
        HoughLines(med, lines, 1, CV_PI / 180, 80);  
          
        //使用迭代器进行图像遍历  
        std::vector<cv::Vec2f>::const_iterator begin = lines.begin();  
        while (begin != lines.end())  
        {  
            float rho = (*begin)[0];//第一个参数为距离ρ  
            float theta = (*begin)[1];//第二个参数为角度θ  
            if (theta<CV_PI / 4. || theta>3.*CV_PI / 4.)//垂直线    
            {  
                //线与第1行的交点  
                cv::Point pt1(0, (rho/sin(theta)));  
                //线与最后一行的交点  
                cv::Point pt2(dst.rows,(rho-dst.rows*cos(theta))/sin(theta));  
                //在原图src中绘制线  
                cv::line(src, pt1, pt2, cv::Scalar(0,255,255), 1);  
            }  
            else//水平线  
            {  
                //线与第1列的交点  
                cv::Point pt1( rho /cos(theta),0);  
                //线与最后一列的交点  
                cv::Point pt2((rho - dst.cols*sin(theta)) / cos(theta), dst.cols);  
                //绘制线  
            cv::line(src, pt1, pt2, cv::Scalar(255,0,255), 1);  
            }  
            begin++;  
        }  
      
        imshow("result", dst);  
        imshow("src",src);  
        waitKey(0);  
      
        return 0;  
    }  

//使用霍夫变换检测园
#include<opencv2\opencv.hpp>
#include <iostream> 
using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("002.jpg");
	if (!src.data){ printf("error"); exit(0); };
	Mat dst,src1;
	src1 = imread("002.jpg");
	//边缘检测
	Canny(src, dst, 125, 350);
	//图像平滑
	GaussianBlur(dst, dst, Size(5, 5), 1.5);

	//霍夫圆变换
	vector<Vec3f>circles;
	HoughCircles(dst, circles, CV_HOUGH_GRADIENT, 2, 52, 200, 160, 15, 100);

	std::vector<cv::Vec3f>::const_iterator begin = circles.begin();
	while (begin != circles.end())
	{
		//绘制园
		circle(src, Point((*begin)[0], (*begin)[1]), (*begin)[2], Scalar(255), 2);

		begin++;
	}
	//	imshow("result", dst);
	imshow("src", src);
	imshow("原图", src1);
	waitKey(0);

	return 0;
}

//图像的腐蚀与膨胀
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("001.jpg");
	if(!src.data){ cout << "error"; exit(1); }
	Mat dst,dst2;
	//腐蚀图像
	erode(src, dst, cv::Mat());
	imshow("腐蚀", dst);

	//膨胀图像
	dilate(src, dst2, cv::Mat());
	imshow("膨胀", dst2);

	imshow("原图", src);
	waitKey(0);
	return 0;
}

//开运算与闭运算
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("001.jpg");
	if(!src.data){ cout << "error"; exit(1); }
	Mat closed,opened;
	cv::Mat element5(5, 5, CV_8U, Scalar(1));
	
	//闭运算
	cv::morphologyEx(src, closed, cv::MORPH_CLOSE, element5);

	imshow("闭运算", closed);

	//膨胀图像
	cv::morphologyEx(src, opened, cv::MORPH_OPEN,element5);
	imshow("开运算", opened);

	imshow("原图", src);
	waitKey(0);
	return 0;
}

#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


class Historam1D
{
private:
	int histSize[1];//项的数量
	float hranges[2];//像素的最小及最大值
	const float*ranges[1];
	int channels[1];//仅用到1个通道

public:
	Historam1D()
	{
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] = hranges;
		channels[0] = 0;
	}
	cv::MatND getHistogram(const cv::Mat &image)
	{
		cv::MatND hist;
		//计算直方图
		cv::calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);


		return hist;
	}
};

int main()
{
	Mat src = imread("group.jpg",0);

	Historam1D h;

	cv::MatND histo = h.getHistogram(src);

	for (int i = 0; i < 256; i++ )
		cout << "Vaule" << i << "=" << histo.at<float>(i) << endl;
	
	
	waitKey(0);
	return 0;
}

#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


class Historam1D
{
private:
	int histSize[1];//项的数量
	float hranges[2];//像素的最小及最大值
	const float*ranges[1];
	int channels[1];//仅用到1个通道

public:
	Historam1D()//1D直方图的参数
	{
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] = hranges;
		channels[0] = 0;
	}
	cv::MatND getHistogram(const cv::Mat &image)
	{
		cv::MatND hist;
		//计算直方图
		cv::calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);

		return hist;
	}
	cv::MatND getHistogramImg(const cv::Mat &image)
	{
		//计算直方图
		cv::MatND hist=getHistogram(image);
		//获取最大值与最小值
		//cv::calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
		double maxVal = 0;
		double minVal = 0;
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);
		//显示直方图图像
		cv::Mat histImg(histSize[0], histSize[0],CV_8U, Scalar(255));
		//设置最高点为nbins的90%
		int hpt = static_cast<int>(0.9*histSize[0]);
		//每个条目都绘制一条垂直线
		for (int h = 0; h < histSize[0]; h++)
		{
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			//两点之间绘制一条线
			cv::line(histImg, Point(h, histSize[0]), Point(h, histSize[0] - intensity), Scalar::all(0));
		}
		return histImg;
	}
};

int main()
{
	Mat src = imread("group.jpg",0);

	Historam1D h;
	cv::MatND histo = h.getHistogram(src);

//	for (int i = 0; i < 256; i++ )
//		cout << "Vaule" << i << "=" << histo.at<float>(i) << endl;
	
	namedWindow("Histogram");
imshow("Histogram", h.getHistogramImg(src));//显示直方图
imshow("src", src);

	waitKey(0);
	return 0;
}
#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\gpu\gpu.hpp>
using namespace std;
using namespace cv;
using namespace cv::gpu;
int main()
{
	int iDeviceNum = getCudaEnabledDeviceCount();
	cout << iDeviceNum << endl;

	return 0;
}

#include<iostream>

using namespace std;

int main()
{
	cout << "input a number" << endl;
	int x = 0;
	cin >> x;
	cout << oct << x << endl;//以8进制输出
	cout << dec << x << endl;//以10进制输出
	cout << hex << x << endl;//以16进制输出

	cout << "输入一个bool值" << endl;
	bool y = false;
	cin >> y;
	cout << boolalpha << y << endl;//以bool方式输出到屏幕上

	return 0;
}
//命名空间的使用
#include<iostream>
using namespace std;

namespace A
{
	int x = 1;
	void fun()
	{
		cout << "A" << endl;
	}
}
namespace B
{
	int x = 2;
	void fun()
	{
		cout << "B" << endl;
	}
}

int main()
{
	cout <<A::x<< endl;//使用A内的
	B::fun();//使用B内的

	return 0;
}

//bool值判断求最大值与最小值
#include<iostream>
using namespace std;
namespace CompanyA//由A公司发明的函数
{
	int maxormin(int *arr, int n, bool Max)
	{
		int temp = arr[0];
		if (Max)
		{
			for (int i = 1; i < n; i++)
			{
				if (temp < arr[i])
					temp = arr[i];
			}
		}
		else
		{
			for (int i = 1; i < n; i++)
			{
				if (temp>arr[i])
					temp = arr[i];
			}
		}
		return temp;

	}
}

int main()
{
	int arr[4] = { 3, 5, 2, 7 };
	bool Max=false;
	cin >> Max;
	cout << CompanyA::maxormin(arr, 4, Max) << endl;;

	return 0;
}
//结构体类型的引用
#include<iostream>
using namespace std;

typedef struct
{
	int x;
	int y;
}Coor;
int main()
{
	Coor c1;
	Coor &c = c1;//c变成了c1的别名
	c.x = 10;
	c.y = 20;
	cout << "(" << c1.x << ","<<c1.y<<")" << endl;
	return 0;
}
//指针类型的引用 类型 *&指针引用名=指针
#include<iostream>
using namespace std;
int main()
{
	int a = 10;
	int *p = &a;

	int *&q = p;
	*q = 20;
	cout << a << endl;
	return 0;
}


/*了解线程块的分配，以及线程束，线程全局标号等
  可以看到总线程数为0~127，共有2个线程块，每个线程块包含64个 线程，
  每个线程块内部线程的索引为0~63.一个线程块包含2个线程束（warp）
  （1个warp包括32个线程）
*/
#include<cuda_runtime.h>
#include<conio.h>
#include<stdio.h>
#include<stdlib.h>
#include<device_launch_parameters.h>

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int)*(ARRAY_SIZE))

__global__ void what_is_my_id(unsigned int *const block,
	unsigned int *const thread,
	unsigned int *const warp,
	unsigned int *const calc_thread)
{
	const unsigned int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;//内部线程的索引
	warp[thread_idx] = threadIdx.x / warpSize;
	calc_thread[thread_idx] = thread_idx;
}

int main()
{
	/* 本地开辟4个数组存放我们要计算的内容 */
	unsigned int cpu_block[ARRAY_SIZE];
	unsigned int cpu_thread[ARRAY_SIZE];
	unsigned int cpu_warp[ARRAY_SIZE];
	unsigned int cpu_calc_thread[ARRAY_SIZE];

	//设计线程数为2*64=128个线程
	const unsigned int num_blocks = 2;
	const unsigned int num_threads = 64;

	/* 在GPU上分配同样大小的4个数组 */
	unsigned int * gpu_block;
	unsigned int * gpu_thread;
	unsigned int * gpu_warp;
	unsigned int * gpu_calc_thread;

	cudaMalloc((void**)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_warp, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);

	//执行内核函数
	what_is_my_id << <num_blocks, num_threads >> >(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

	//将GPU运算完的结果复制回本地
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(gpu_block);
	cudaFree(gpu_thread);
	cudaFree(gpu_warp);
	cudaFree(gpu_calc_thread);

	//输出
	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("总线程数%3u-Blocks:%2u-Warp%2u-内部线程数%3u\n",
			cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
	}

	return 0;
}
