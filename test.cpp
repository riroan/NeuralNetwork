#include"Model.h"
using namespace std;

int main()
{
	Model model;
	matrix m(10, 10);
	m.assign_random(0.0, 1.0);
	matrix m1(10, 10);
	m1.assign_random(0.0, 1.0);
	matrix m2(10, 10);
	m2.assign_random(0.0, 1.0);

	model.addConvolution(10, 10, 3, 3, 3, 16, 1, 1, "same");
	model.maxPool(2);
	model.addConvolution(2, 2, 32, "valid");
	model.maxPool(2);
	model.addAffine(10, RELU);
	model.addAffine(5, RELU);
	model.addAffine(2, IDENTITY);

	Vector<double> v(2);
	v.assign_random(4.0, 4.0);
	v[0] = 2.0;

	model.setInput(m);
	model.setInput(m1);
	model.setInput(m2);

	for (int i = 0; i < 10000; i++)
	{
		model.forwardPropagation();

		model.backPropagation(v);
		if(i%100==0)
			cout << i << " : "<< "Error : " << model.Error << ", output : " << model.getOutput() << endl << endl;
	}
}