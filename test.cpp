#include"Model.h"
using namespace std;

int main()
{
	Model model;
	matrix m(5, 5);
	matrix m1(5, 5);
	matrix m2(5, 5);
	m.assign_random(0.0, 1.0);
	m1.assign_random(0.0, 1.0);
	m2.assign_random(0.0, 1.0);
	Vector<double> y(2, 2.0);

	model.addConvolution(5, 5, 3, 3, 3, 2);
	model.addAffine(2, RELU);
	model.setInput(m);
	model.setInput(m1);
	model.setInput(m2);

	//model.layer_C[0]->kernel[0][0].filter.print();
	//cout << model.layer_C[0]->kernel[0][0].bias;
	//cout << endl;
	//model.layer_C[0]->kernel[0][1].filter.print();
	//cout << model.layer_C[0]->kernel[0][1].bias;
	//cout << endl;
	//model.layer_C[0]->kernel[0][2].filter.print();
	//cout << model.layer_C[0]->kernel[0][2].bias;
	//cout << endl;
	//model.layer_C[0]->kernel[1][0].filter.print();
	//cout << model.layer_C[0]->kernel[1][0].bias;
	//cout << endl;
	//model.layer_C[0]->kernel[1][1].filter.print();
	//cout << model.layer_C[0]->kernel[1][1].bias;
	//cout << endl;
	//model.layer_C[0]->kernel[1][2].filter.print();
	//cout << model.layer_C[0]->kernel[1][2].bias;
	//cout << endl;
	//cout << model.layer_A[0]->w << endl;
	//cout << model.layer_A[0]->b << endl;

	model.forwardPropagation();

	model.backPropagation(y);
}