#include"Model.h"
using namespace std;

int main()
{
	Model model;
	model.addConvolution(5, 5, 2, 2, 1, 2, 1, 1);
	model.maxPool(2);
	model.addAffine(2, RELU);

	matrix m(5, 5);
	m.assign_random(0.0, 1.0);

	Vector<double> v(2);
	v.assign_random(4.0, 4.0);
	v[0] = 2.0;

	model.setInput(m);

	for (int i = 0; i < 1000; i++)
	{
		cout << i << " : ";
		model.forwardPropagation();

		model.backPropagation(v);
			cout << "Error : " << model.Error << ", output : " << model.getOutput() << endl << endl;
		//if (i > 73)
			//getchar();
	}
}