// using my model
#include<iostream>
#include<vector>
#include<ctime>
#include"Model.h"
#include"Network.h"
#include<chrono>
#include<windows.h>
using namespace std;

int main()
{
	Vector<Vector<double>> x(4);
	for (int i = 0; i < 4; i++)
		x[i].resize(2);
	x[0][0] = 0.0; x[0][1] = 0.0;
	x[1][0] = 1.0; x[1][1] = 0.0;
	x[2][0] = 0.0; x[2][1] = 1.0;
	x[3][0] = 1.0; x[3][1] = 1.0;

	Vector<Vector<double>> y(4);
	for (int i = 0; i < 4; i++)
		y[i].resize(1);
	y[0][0] = 0.0;
	y[1][0] = 1.0;
	y[2][0] = 1.0;
	y[3][0] = 0.0;

	int cnt = 0;
	
	srand(time(NULL));

	int correct = 0;

	for (int r = 0; r < 1; r++)
	{
		Model model;

		model.addAffine(2, 2, RELU);
		model.addAffine(1, RELU);

		Vector<Vector<double>> out(4);

		for (int i = 0; i < 2000; i++)
		{
			int ix = i % 4;

			model.setInput(x[ix]);

			model.forwardPropagation();

			model.backPropagation(y[ix]);

			cout << endl << endl;
			system("clear");
			for (int k = 0; k < model.num_of_affine; k++)
			{
				cout << "w" << endl;
				model.layer_A[k]->w.print();
				cout << "b" << endl;
				model.layer_A[k]->b.print();
				cout << "out_gradient" << endl;
				model.layer_A[k]->out_grad.print();
				cout << "layer_act" << endl;
				model.layer_A[k]->output.print();
				cout << endl;
			}
			cout << "output" << endl;
			model.getOutput().print();


			if (i % 10000 < 4)
			{
				cout << "Error " << i << " : " << model.Error << endl;
			}
			getchar();
			//getchar();
		}

		for (int e = 0; e < 4; e++)
		{
			out[e].resize(1);
			model.setInput(x[e]);
			model.forwardPropagation();

			out[e] = model.getOutput();

			out[e].print();
		}

		printf("\n");

		if (out[0][0] < 0.01&&out[1][0] >= 0.99&&out[2][0] >= 0.99&&out[3][0] <= 0.01)
			correct++;

		cout << model.Error << endl;
	}
	printf("%lf", (double)correct / 10.0);
}