#include<iostream>
#include<vector>
#include<ctime>
#include"Network.h"
using namespace std;

int main()
{
	Network nn(2, 1, 1);
	vector<double> x1{ 0,0 };
	vector<double> x2{ 0,1 };
	vector<double> x3{ 1,0 };
	vector<double> x4{ 1,1 };

	vector<double> y1{ 0 };
	vector<double> y2{ 1 };
	vector<double> y3{ 1 };
	vector<double> y4{ 0 };

	for (int i = 0; i < 1000; i++)
	{
		nn.setInput(x1);
		nn.feedForward();
		nn.backPropagation(y1);

		nn.setInput(x2);
		nn.feedForward();
		nn.backPropagation(y2);

		nn.setInput(x3);
		nn.feedForward();
		nn.backPropagation(y3);

		nn.setInput(x4);
		nn.feedForward();
		nn.backPropagation(y4);
	}

	nn.setInput(x1);
	nn.feedForward();
	cout << (nn.getOutput())[0] << endl;

	nn.setInput(x2);
	nn.feedForward();
	cout << (nn.getOutput())[0] << endl;

	nn.setInput(x3);
	nn.feedForward();
	cout << (nn.getOutput())[0] << endl;

	nn.setInput(x4);
	nn.feedForward();
	cout << (nn.getOutput())[0] << endl;
}