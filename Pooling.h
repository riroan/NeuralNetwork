#pragma once
#include"matrix.h"
#include<tuple>

class Pooling
{
public:
	short x;
	short stride;

	Pooling(const int& _x);
	matrix maxPool(const matrix& input, matrix& gradient);
	matrix getGrad(const matrix& out_grad);
};