//#include<iostream>
//#include<vector>
//#include<random>
//using namespace std;
//
//double f1(double x1, double x2)
//{
//	return x1 * x1 + x2 * x2;
//}
//
//double f2(double x)
//{
//	return 2 * x / (x*x + 1);
//}
//
//double gradient(double (*f)(double, double), double x1, double x2)
//{
//	double h = 1e-4;
//	double grad = 0.0;
//	grad = (f(x1 + h, x2 + h) - f(x1 - h, x2 - h)) / (2 * h);
//	return grad;
//}
//
//double gradient(double(*f)(double), double x1)
//{
//	double h = 1e-4;
//	double grad = 0.0;
//	grad = (f(x1 + h) - f(x1 - h)) / (2 * h);
//	return grad;
//}
//
//double gradient_descent(double(*f)(double), double init_x)
//{
//	double x = init_x;
//
//	for (int i = 0; i < 1000; i++)
//	{
//		double grad = gradient(f, x);
//		x -= 0.1*grad;
//		cout << i << " " << x << endl;
//	}
//	return x;
//}
//
//default_random_engine generator;
//
//void f()
//{
//	normal_distribution<double> distribution(0.0, 1.0);
//	for (int i = 0; i < 5; i++)
//		cout << distribution(generator)<<" ";
//}
//
//int main()
//{
//	f();
//	cout << endl;
//	f();
//}