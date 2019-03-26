//#include<iostream>
//#include<vector>
//#include<random>
//#include<tuple>
//#include<windows.h>
//#include<utility>
//#include"Convolution1D.h"
//#include"Convolution2D.h"
//#include"Affine.h"
////#include"matrix.h"
//#pragma warning(disable:4996)
//#define EPOCH 500
//
//double f1(double x1, double x2)
//{
//	return x2 * x2 + x1 * x1 / 100;
//	//return (x1 * x1) / 100.0 + x2 * x2;
//}
//
//auto gradient(double(*f)(double, double), double x, double y)
//{
//	double h = 1e-4;
//	double grad_x = 0.0;
//	double grad_y = 0.0;
//	grad_x = (f(x + h, y) - f(x, y)) / h;
//	grad_y = (f(x, y + h) - f(x, y)) / h;
//
//	return tuple<double, double>(grad_x, grad_y);
//}
//
//double gradient_descent_SGD(double(*f)(double, double), double init_x, double init_y)
//{
//	double x = init_x;
//	double y = init_y;
//
//	FILE * fp = fopen("SGD.txt", "w+");
//	for (int i = 0; i < EPOCH; i++)
//	{
//		auto[grad_x, grad_y] = gradient(f, x, y);
//		x -= 0.1*grad_x;
//		y -= 0.1*grad_y;
//		//if (i > 9000)
//		cout << i << " : " << x << ", " << y << endl;
//
//		fprintf(fp, "%lf %lf\n", x, y);
//	}
//	fclose(fp);
//	return x;
//}
//
//double gradient_descent_momentum(double(*f)(double, double), double init_x, double init_y)
//{
//	double x = init_x;
//	double y = init_y;
//
//	static double v_x = 0.0;
//	static double v_y = 0.0;
//
//	FILE * fp = fopen("momentum.txt", "w+");
//	for (int i = 0; i < EPOCH; i++)
//	{
//		auto[grad_x, grad_y] = gradient(f, x, y);
//
//		v_x = v_x * 0.5 - 0.1*grad_x;
//		v_y = v_y * 0.5 - 0.1*grad_y;
//
//		x += v_x;
//		y += v_y;
//		//if (i > 9000)
//		cout << i << " : " << x << ", " << y << endl;
//		fprintf(fp, "%lf %lf\n", x, y);
//	}
//	fclose(fp);
//	return x;
//}
//
//double gradient_descent_AdaGrad(double(*f)(double, double), double init_x, double init_y)
//{
//	double x = init_x;
//	double y = init_y;
//
//	static double h_x = 0.0;
//	static double h_y = 0.0;
//
//	FILE * fp = fopen("AdaGrad.txt", "w+");
//	for (int i = 0; i < EPOCH; i++)
//	{
//		auto[grad_x, grad_y] = gradient(f, x, y);
//
//		h_x += grad_x * grad_x;
//		h_y += grad_y * grad_y;
//
//		x -= 1 / sqrt(h_x + 1e-7)*grad_x;
//		y -= 1 / sqrt(h_y + 1e-7)*grad_y;
//		//if (i > 9000)
//		cout << i << " : " << x << ", " << y << endl;
//
//		fprintf(fp, "%lf %lf\n", x, y);
//	}
//	fclose(fp);
//	return x;
//}
//
//double gradient_descent_Adam(double(*f)(double, double), double init_x, double init_y)
//{
//	double x = init_x;
//	double y = init_y;
//
//	int t = 0;
//	double s_x = 0.0;
//	double r_x = 0.0;
//	double s_y = 0.0;
//	double r_y = 0.0;
//
//	double p1 = 0.9;
//	double p2 = 0.999;
//
//	double ds_x;
//	double ds_y;
//
//	double dr_x;
//	double dr_y;
//
//	FILE * fp = fopen("Adam.txt", "w+");
//	for (int i = 0; i < EPOCH; i++)
//	{
//		auto[grad_x, grad_y] = gradient(f, x, y);
//
//		t++;
//
//		s_x = p1 * s_x + (1 - p1)* grad_x;
//		s_y = p1 * s_y + (1 - p1)* grad_y;
//
//		r_x = p2 * r_x + (1 - p2)*grad_x*grad_x;
//		r_y = p2 * r_y + (1 - p2)*grad_y*grad_y;
//
//		ds_x = s_x / (1 - pow(p1, t));
//		ds_y = s_y / (1 - pow(p1, t));
//
//		dr_x = r_x / (1 - pow(p2, t));
//		dr_y = r_y / (1 - pow(p2, t));
//
//		double dx = -0.1*ds_x / (sqrt(dr_x) + 1e-8);
//		double dy = -0.1*ds_y / (sqrt(dr_y) + 1e-8);
//
//		x += dx;
//		y += dy;
//		//if (i > 9000)
//		cout << i << " : " << x << ", " << y << endl;
//
//		fprintf(fp, "%lf %lf\n", x, y);
//	}
//	fclose(fp);
//	return x;
//}
//
//double gradient_descent_RMS(double(*f)(double, double), double init_x, double init_y)
//{
//	double x = init_x;
//	double y = init_y;
//
//	double r_x = 0.0;
//	double r_y = 0.0;
//
//	double p = 0.9999;
//
//	FILE * fp = fopen("RMS.txt", "w+");
//	for (int i = 0; i < EPOCH; i++)
//	{
//		auto[grad_x, grad_y] = gradient(f, x, y);
//
//		r_x = p * r_x + (1 - p)*grad_x*grad_x;
//		r_y = p * r_y + (1 - p)*grad_y*grad_y;
//
//		double dx = -0.01 / sqrt(r_x + 1e-6)*grad_x;
//		double dy = -0.01 / sqrt(r_y + 1e-6)*grad_y;
//
//		x += dx;
//		y += dy;
//		//if (i > 9000)
//		cout << i << " : " << x << ", " << y << endl;
//
//		fprintf(fp, "%lf %lf\n", x, y);
//	}
//	fclose(fp);
//	return x;
//}
//
//vector<double> f50()
//{
//	std::vector<double> ret(500000);
//	for (int i = 0; i < 500000; i++)
//		ret[i] = i;
//	return ret;
//}
//
//void f51(vector<double>& v)
//{
//	v.resize(500000);
//	for (int i = 0; i < 500000; i++)
//		v[i] = i;
//}
//
//void f52(tuple<double, double> a)
//{
//}
//
//void f(int number);
//
//int main()
//{
//	cout << "==========SGD==========" << endl;
//	gradient_descent_SGD(f1, 5.0, 5.0);
//
//	cout << endl << "==========momentum==========" << endl;
//	gradient_descent_momentum(f1, 5.0, 5.0);
//
//	cout << endl << "==========AdaGrad==========" << endl;
//	gradient_descent_AdaGrad(f1, 5.0, 5.0);
//
//	cout << endl << "==========Adam==========" << endl;
//	gradient_descent_Adam(f1, 5.0, 5.0);
//
//	cout << endl << "==========RMS==========" << endl;
//	gradient_descent_RMS(f1, 5.0, 5.0);
//
//	//f(5000000);
//
//	//LARGE_INTEGER BeginTime, EndTime, Frequency;
//
//	//vector<double> v1(1000);
//	//double * v2 = new double[1000];
//	//for (int i = 0; i < 1000; i++)
//	//{
//	//	v1[i] = i;
//	//	v2[i] = i;
//	//}
//
//	//double sum = 0.0;
//
//	//QueryPerformanceFrequency(&Frequency);
//
//	//QueryPerformanceCounter(&BeginTime);
//
//	//for (int i = 0; i < 1000; i++)
//	//	sum += v1[i];
//
//	//QueryPerformanceCounter(&EndTime);
//	//cout << "Outside : " << double(EndTime.QuadPart - BeginTime.QuadPart) / Frequency.QuadPart * 1000 << endl;
//
//	//QueryPerformanceCounter(&BeginTime);
//
//	//for (int i = 0; i < 1000; i++)
//	//	sum += v2[i];
//
//	//QueryPerformanceCounter(&EndTime);
//	//cout << "Outside : " << double(EndTime.QuadPart - BeginTime.QuadPart) / Frequency.QuadPart * 1000 << endl;
//
//
//	//Convolution1D c(1, 3, 0, "same");
//	//vector<double> x(10);
//	//for (double& i : x)
//	//	i = rand() / (double)RAND_MAX;
//
//	//cout << endl;
//	//c.setInput(x);
//	//c.feedForward();
//
//	//for (auto i : c.output)
//	//	cout << " " << i;
//
//	//Convolution2D conv(1, 3, 3, 0);
//	//matrix input(5, 5);
//	//input.assign_random(0.1, 1.0);
//
//	//conv.setInput(input);
//	//conv.input.print();
//	//printf("\n");
//	//conv.feedForward();
//	//conv.output.print();
//	
//	//Convolution2D c(1, 3, 3, 0);
//	//matrix m(5, 5);
//	//m.print();
//	//c.setInput(m);
//	//c.feedForward();
//
//	//Affine layer(1, 1);
//	//Vector<double> x(1, 1.0);
//	//Vector<double> y(1, 3.0);
//
//	//layer.input = x;
//	//for (int i = 0; i < 1000; i++)
//	//{
//	//	layer.feedForward();
//	//	layer.getMSE(y);
//	//	layer.update_weight();
//	//	cout << layer.output[0] << endl;
//	//}
//
//	//Vector<double> v(5, 2);
//	//v.print();
//	//v.push(3);
//	//v.print();
//
//	//Convolution2D c(5, 5, 1, 3, 3, 0, RELU);
//	//matrix input(5, 5);
//	//matrix filter(3, 3);
//	//for (int i = 0; i < 5 * 5; i++)
//	//	input[i] = static_cast<double>(i) / 10.0;
//	//for (int i = 0; i < 9; i++)
//	//	filter[i] = (double)i / 10.0;
//
//	//c.setInput(input);
//	//c.filter = filter;
//	//c.bias = 0.0;
//	//matrix desire(3, 3);
//	//desire.assign_random(0.5, 0.5);
//	//for (int i = 0; i <500; i++)
//	//{
//	//	c.feedForward();
//	//	c.backPropagation(desire);
//	//	c.output.print();
//	//	cout << endl;
//	//}
//	//f(1000000);
//}
//
//random_device generator;
//
//void f(int number)
//{
//	normal_distribution<double> distribution(0.0,1.0);
//
//	int cnt[10] = { 0 };
//
//	double * x = new double[number];
//	double mean = 0.0;
//	for (int i = 0; i < number; i++)
//	{
//		x[i] = distribution(generator)/5;
//		mean += x[i];
//		if (x[i] >= 0.0&&x[i] < 0.1)
//			cnt[0]++;
//		else if (x[i] < 0.2)
//			cnt[1]++;
//		else if (x[i] < 0.3)
//			cnt[2]++;
//		else if (x[i] < 0.4)
//			cnt[3]++;
//		else if (x[i] < 0.5)
//			cnt[4]++;
//		else if (x[i] < 0.6)
//			cnt[5]++;
//		else if (x[i] < 0.7)
//			cnt[6]++;
//		else if (x[i] < 0.8)
//			cnt[7]++;
//		else if (x[i] < 0.9)
//			cnt[8]++;
//		else
//			cnt[9]++;
//	}
//	mean /= number;
//
//	double sigma = 0.0;
//
//	for (int i = 0; i < number; i++)
//		sigma += (x[i] - mean)*(x[i] - mean);
//	sigma /= number;
//
//	cout << "mean : " << mean << ", " << "sigma : " << sqrt(sigma) << endl;
//	for (int i = 0; i < 10; i++)
//		cout << " " << cnt[i];
//}