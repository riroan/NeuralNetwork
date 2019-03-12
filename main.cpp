//// using my model
//#include<iostream>
//#include<vector>
//#include<ctime>
//#include"Model.h"
//#include"Network.h"
//#include<chrono>
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
//		//Network nn(2, 1, 1);
//		Model model;
//
//		model.addAffine(2, 1, RELU);
//
//		model.addAffine(1);
//
//		Vector<Vector<double>> out(4);
//
//		for (int i = 0; i < 10000; i++)
//		{
//			//int ix = rand() % 4;
//			int ix = i % 4;
//
//			model.setInput(x[ix]);
//
//			//auto sta = chrono::steady_clock::now();
//
//			model.forwardPropagation();
//
//			//auto dur = chrono::steady_clock::now() - sta;
//			//cout << dur.count() << endl;
//
//			model.backPropagation(y[ix]);
//
//			//model.getOutput().print();
//
//			//nn.setInput(x[ix]);
//			//nn.feedForward();
//
//			//nn.backPropagation(y[ix]);
//
//			//cout <<"Error : " << model.Error << endl;
//			//getchar();
//		}
//
//		for (int e = 0; e < 4; e++)
//		{
//			out[e].resize(1);
//			//nn.setInput(x[e]);
//			//nn.feedForward();
//			model.setInput(x[e]);
//			model.forwardPropagation();
//
//			out[e] = model.getOutput();
//
//			//out[e] = nn.getOutput();
//			out[e].print();
//		}
//		printf("\n");
//
//
//		if (out[0][0] < 0.0001&&out[1][0] >= 0.9999&&out[2][0] >= 0.9999&&out[3][0] <= 0.0001)
//			correct++;
//	}
//	printf("%lf", (double)correct / 10.0);
//}