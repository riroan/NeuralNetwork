#pragma once
#include"matrix.h"

class Convolution2D
{
public:
	matrix input;
	matrix filter;
	matrix output;
	matrix gradient;
	bool   pad_type;
	int	   padding;
	int    stride;
	double learning_rate;
	double bias;

	Convolution2D(const int& _stride, const int& f_w, const int& f_h, const int& _padding, const char * _pad_type = "none");
	void setInput(const matrix& _input);
	void feedForward();
	void backPropagation(const matrix& out_grad);
	void getGrad(matrix out_grad);
	void update_weight(matrix out_grad);
};