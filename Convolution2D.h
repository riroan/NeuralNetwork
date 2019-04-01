#pragma once
#include"matrix.h"

class Convolution2D
{
public:
	matrix input;
	matrix filter;
	matrix output;
	matrix gradient;
	int	   padding;
	int    stride;
	int	   layer_act;
	double learning_rate;
	double bias;

	Convolution2D() {}
	Convolution2D(const int& _row, const int& _col, const int& f_w, const int& f_h, const int& _layer_act = RELU, const int& _stride = 1, const char * _pad_type = "valid");
	Convolution2D(const int& f_w, const int& f_h, const int& _layer_act = RELU, const int& _stride = 1, const char * _pad_type = "valid");
	void setInput(const matrix& _input);
	void feedForward();

	// only padding is 0 and stride is 1
	void backPropagation(const matrix& out_grad);
	void getGrad(matrix out_grad);
	void update_weight(matrix out_grad);

	void apply_sigmoid();
	void apply_ReLU();
	void apply_LReLU();

	inline double sigmoid(const double& x) { return 1.0 / (1.0 + exp(-x)); }
	inline double ReLU(const double& x) { return x > 0.0 ? x : 0.0; }
	inline double LReLU(const double& x) { return x > 0.0 ? x : 1e-2*x; }

	inline double grad_sigmoid(const double& y) { return (1.0 - y)*y; }
	inline double grad_ReLU(const double& y) { return y > 0 ? 1.0 : 0.0; }
	inline double grad_LReLU(const double& y) { return y > 0 ? 1.0 : 1e-2; }
};