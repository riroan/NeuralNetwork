#include "Affine.h"

Affine::Affine(const matrix& _W, const vector<double>& _b)
	:dW(0.0),db(0.0)
{
	W = _W;
	b = _b;
}

vector<double> Affine::forward(const vector<double>& _x)
{
	x = _x;
	vector<double> ret = W * x;
	ret = vsum(ret, b);

	return ret;
}