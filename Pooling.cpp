#include"Pooling.h"

Pooling::Pooling(const int& _x)
	:x(_x)
{
	stride = x;
}

matrix Pooling::maxPool(const matrix& input, matrix& gradient)
{
	gradient.resize(input.row, input.col);
	auto f = [&](const int& x, const int& y)
	{
		double max = -1e100;
		int px = 0, py = 0;
		for (short i = y; i < y + stride; i++)
			for (short j = x; j < x + stride; j++)
			{
				if (max < input.getValue(i, j))
				{
					max = input.getValue(i, j);
					px = j;
					py = i;
				}
			}
		for (short i = y; i < y + stride; i++)
			for (short j = x; j < x + stride; j++)
				if (!(i == py && j == px))
					gradient.getValue(i, j) = 0;
				else
					gradient.getValue(i, j) = 1;
		return std::tuple(max, px, py);
	};
	int y = input.row, x = input.col;
	matrix ret(y / stride, x / stride);
	for (int i = 0; i < y; i += stride)
		for (int j = 0; j < x; j += stride)
		{
			auto[temp, px, py] = f(j, i);
			ret.getValue(i / stride, j / stride) = temp;
		}
	return ret;
}

matrix Pooling::getGrad(const matrix& out_grad)
{
	matrix ret(out_grad.row*stride, out_grad.col * stride);
	for (int i = 0; i < ret.row; i += stride)
		for (int j = 0; j < ret.col; j += stride)
			for (short k = i; k < i + stride; k++)
				for (short l = j; l < j + stride; l++)
					ret.getValue(k, l) = out_grad.getValue(i / stride, j / stride);
	return ret;
}