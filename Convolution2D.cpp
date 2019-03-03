#include"Convolution2D.h"

Convolution2D::Convolution2D(const int& _stride, const int& f_w, const int& f_h, const int& _padding, const char * _pad_type)
	:padding(_padding), stride(_stride), filter(f_h, f_w)
{
	filter.assign_random(0.0, 1.0);

	if (strcmp(_pad_type, "same") == 0)
		pad_type = true;
	else
		pad_type = false;
}

void Convolution2D::setInput(const matrix& _input)
{
	input = _input;
	if (pad_type)
	{
		padding = (stride * (input.row - 1) - input.row + filter.row) / 2;
	}
	int out_row = (input.row + 2 * padding - filter.row) / stride + 1;
	int out_col = (input.col + 2 * padding - filter.col) / stride + 1;
	output.resize(out_row, out_col);
	output.assign_random(0.0, 0.0);
}

void Convolution2D::feedForward()
{
	for (int i = -padding, x = 0; i < input.row + padding - filter.row + 1; i += stride, x++)
		for (int j = -padding, y = 0; j < input.col + padding - filter.col + 1; j += stride, y++)
			output.getValue(x, y) = input.Convolution(filter, i, j);
}