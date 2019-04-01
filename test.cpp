#include"Model.h"
#include<chrono>

int main()
{
	Model model;
	matrix m(150, 150);
	matrix m1(150, 150);
	matrix m2(150, 150);
	m.assign_random(0.0, 1.0);
	m1.assign_random(0.0, 1.0);
	m2.assign_random(0.0, 1.0);

	model.addConvolution(150, 150, 3, 3, 3, 32);
	model.addConvolution(3, 3, 64, "valid");
	model.addAffine(512, RELU);
	model.addAffine(1, RELU);
	model.setInput(m);
	model.setInput(m1);
	model.setInput(m2);
	using namespace std;
	auto sta = std::chrono::steady_clock::now();

	model.forwardPropagation();

	chrono::duration<double> dur = chrono::steady_clock::now() - sta;
	cout << dur.count() << endl;
	model.getOutput().print();
}