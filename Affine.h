#pragma once
#include"matrix.h"


class Affine
{
public:
	Vector<double> input;
	Vector<double> output;
	int			   layer_act;
	int			   num_input;
	int			   num_output;
	matrix		   w;
	Vector<double> b;
	Vector<double> gradient;	// gradient of x not w
	Vector<double> out_grad;	// gradient of next layer
	double		   learning_rate;

	Affine(const int& _num_input, const int& _num_output, const int& activation = RELU);

	void feedForward();
	void getMSE(const Vector<double>& y_t);
	void setGrad(const Vector<double>& grad);
	void getGrad();
	void update_weight();

	void apply_sigmoid(Vector<double>& v);
	void apply_ReLU(Vector<double>& v);
	void apply_LReLU(Vector<double>& v);
	void apply_softmax(Vector<double>& v);

	inline double sigmoid(const double& x) { return 1.0 / (1.0 + exp(-x)); }
	inline double ReLU(const double& x) { return x > 0.0 ? x : 0.0; }
	inline double LReLU(const double& x) { return x > 0.0 ? x : 1e-2*x; }

	inline double grad_sigmoid(const double& y) { return (1.0 - y)*y; }
	inline double grad_ReLU(const double& y) { return y > 0 ? 1.0 : 0.0; }
	inline double grad_LReLU(const double& y) { return y > 0 ? 1.0 : 1e-2; }
};