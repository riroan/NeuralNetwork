#include"Model.h"
#include"Pooling.h"
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
	Vector<double> y(2, 10.0);
	
	model.addConvolution(5, 5, 3, 3, 3, 2, 1, 1, "same");
	model.addConvolution(2, 2);
	model.maxPool(2);
	model.addAffine(2, RELU);
	model.setInput(m);
	model.setInput(m1);
	model.setInput(m2);

	for (int i = 0; i < 10000; i++)
	{
		model.forwardPropagation();

		model.backPropagation(y);
		if (i > 9000)
		{
			cout << model.getOutput() << endl;
			cout << " Error " << i << " : " << model.Error << endl;
		}
	}



	//Model model;
	//model.addConvolution(5, 5, 2, 2);
	////model.maxPool(2);
	//model.addAffine(2, LRELU);

	//matrix m(5, 5);
	//m.assign_random(0.0, 2.0);
	//Vector<double> y_t(2, 1);
	//model.setInput(m);

	//for (int i = 0; i < 10000; i++)
	//{
	//	model.forwardPropagation();
	//	model.backPropagation(y_t);
	//	cout << model.getOutput() << endl;
	//	cout << " Error : " << model.Error << endl;
	//}
}