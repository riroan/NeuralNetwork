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
	model.addConvolution(3, 3, 64);
	model.addAffine(10, RELU);
	model.addAffine(5, RELU);
	model.addAffine(6, IDENTITY);

	Vector<double> v(6);
	v.assign_random(1.0, 1.0);
	v[0] = 2.0;
	v[1] = 4.0;
	v[2] = 8.0;
	v[3] = 16.0;
	v[4] = 32.0;

	model.setInput(m);
	model.setInput(m1);
	model.setInput(m2);

	auto sta = std::chrono::steady_clock::now();

	for (int i = 0; i < 1000; i++)
	{

		//auto sta = std::chrono::steady_clock::now();

		model.forwardPropagation();

		//std::chrono::duration <double> dur = std::chrono::steady_clock::now() - sta;
		//std::cout << dur.count() << std::endl;

		model.backPropagation(v);


		if (i % 1 == 0)
			cout << i << " : " << "Error : " << model.Error << ", output : " << model.getOutput() << endl << endl;
		//getchar();
	}
	std::chrono::duration <double> dur = std::chrono::steady_clock::now() - sta;
	std::cout << dur.count() << std::endl;
}