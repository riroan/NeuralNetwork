#pragma once
#include "matrix.h"

class Affine
{
public:
	matrix		   W;
	vector<double> b;
	vector<double> x;
	double		   dW;
	double		   db;

	Affine(const matrix& _W, const vector<double>& _b);
	vector<double> forward(const vector<double>& _x);
	double backward(double dout);
};