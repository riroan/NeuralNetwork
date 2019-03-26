//#include"Model.h"
//#pragma warning(disable:4996)
//using namespace std;
//
//int main()
//{
//	Model model;
//	model.addAffine(4, 2, RELU);
//	model.addAffine(2, RELU);
//
//	Vector<double> x(4, 1.0);
//	Vector<double> target(2);
//	Vector<double> bias(2, 0);
//
//	target[0] = 2; target[1] = 3;
//
//	matrix w1(2, 4);
//	matrix w2(2, 2);
//	w1.getValue(0, 0) = 0.1;	w1.getValue(1, 0) = 0.2;
//	w1.getValue(0, 1) = 0.3;	w1.getValue(1, 1) = 0.4;
//	w1.getValue(0, 2) = 0.5;	w1.getValue(1, 2) = 0.6;
//	w1.getValue(0, 3) = 0.7;	w1.getValue(1, 3) = 0.8;
//
//	w2.getValue(0, 0) = 0.4;	w2.getValue(1, 0) = 0.5;
//	w2.getValue(0, 1) = 0.3;	w2.getValue(1, 1) = 0.2;
//
//	model.layer_A[0]->b = bias;
//	model.layer_A[1]->b = bias;
//	model.layer_A[0]->w = w1;
//	model.layer_A[1]->w = w2;
//
//	model.setInput(x);
//	model.forwardPropagation();
//	model.backPropagation(target);
//
//	model.layer_A[0]->w.print();
//	model.layer_A[1]->w.print();
//	model.layer_A[0]->b.print();
//	model.layer_A[1]->b.print();
//}