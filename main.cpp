//// using my model
//#include<iostream>
//#include<vector>
//#include<ctime>
//#include"Model.h"
//#include"Network.h"
//#include<chrono>
//#include<windows.h>
//using namespace std;
//
//int main()
//{
//	Vector<Vector<double>> x(4);
//	for (int i = 0; i < 4; i++)
//		x[i].resize(2);
//	x[0][0] = 0.0; x[0][1] = 0.0;
//	x[1][0] = 1.0; x[1][1] = 0.0;
//	x[2][0] = 0.0; x[2][1] = 1.0;
//	x[3][0] = 1.0; x[3][1] = 1.0;
//
//	Vector<Vector<double>> y(4);
//	for (int i = 0; i < 4; i++)
//		y[i].resize(1);
//	y[0][0] = 0.0;
//	y[1][0] = 1.0;
//	y[2][0] = 1.0;
//	y[3][0] = 0.0;
//
//	int cnt = 0;
//	
//	srand(time(NULL));
//
//	int correct = 0;
//
//	for (int r = 0; r < 10; r++)
//	{
//		Model model;
//
//		model.addAffine(2, 10, LRELU);
//		model.addAffine(10, LRELU);
//		model.addAffine(10, LRELU);
//		model.addAffine(10, LRELU);
//		model.addAffine(1, LRELU);
//
//		Vector<Vector<double>> out(4);
//
//		for (int i = 0; i < 10000; i++)
//		{
//			int ix = i % 4;
//
//			model.setInput(x[ix]);
//
//			model.forwardPropagation();
//
//			model.backPropagation(y[ix]);
//
//			if (i % 1000 < 4)
//				cout << "Error " << i << " : " << model.Error << endl;
//		}
//
//		for (int e = 0; e < 4; e++)
//		{
//			out[e].resize(1);
//			model.setInput(x[e]);
//			model.forwardPropagation();
//
//			out[e] = model.getOutput();
//
//			out[e].print();
//		}
//
//		printf("\n");
//
//		if (out[0][0] < 0.01&&out[1][0] >= 0.99&&out[2][0] >= 0.99&&out[3][0] <= 0.01)
//			correct++;
//
//		cout << model.Error << endl;
//	}
//	printf("%lf", (double)correct / 10.0);
//}