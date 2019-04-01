#include"Convolution2D.h"

Convolution2D::Convolution2D(const int& _row, const int& _col, const int& f_w, const int& f_h, const int& _layer_act, const int& _stride, const char * _pad_type)
	:input(_row, _col), stride(_stride), filter(f_h, f_w), learning_rate(0.01), gradient(_row, _col)
{
	layer_act = _layer_act;

	filter.assign_random(0.0, 1.0);

	if (strcmp(_pad_type, "same") == 0)
		padding = (stride * (input.row - 1) - input.row + filter.row) / 2;
	else
		padding = 0;

	bias = (double)rand() / RAND_MAX;

	int out_row = (input.row + 2 * padding - filter.row) / stride + 1;
	int out_col = (input.col + 2 * padding - filter.col) / stride + 1;
	output.resize(out_row, out_col);
	output.assign_random(0.0, 0.0);
}

Convolution2D::Convolution2D(const int& f_w, const int& f_h, const int& _layer_act, const int& _stride, const char * _pad_type)
{
	layer_act = _layer_act;

	filter.assign_random(0.0, 1.0);

	if (strcmp(_pad_type, "same") == 0)
		padding = (stride * (input.row - 1) - input.row + filter.row) / 2;
	else
		padding = 0;

	bias = (double)rand() / RAND_MAX;

	int out_row = (input.row + 2 * padding - filter.row) / stride + 1;
	int out_col = (input.col + 2 * padding - filter.col) / stride + 1;
	output.resize(out_row, out_col);
	output.assign_random(0.0, 0.0);
}

void Convolution2D::setInput(const matrix& _input)
{
	assert(_input.row == input.row&& _input.col == input.col);
	input = _input;
}

void Convolution2D::feedForward()
{
	for (int i = -padding, x = 0; i < input.row + padding - filter.row + 1; i += stride, x++)
		for (int j = -padding, y = 0; j < input.col + padding - filter.col + 1; j += stride, y++)
			output.getValue(x, y) = input.Convolution(filter, i, j);
	output += bias;
	if (layer_act == RELU)
		apply_ReLU();
	else if (layer_act == LRELU)
		apply_LReLU();
	else if (layer_act == SIGMOID)
		apply_sigmoid();
}

// only padding is 0 and stride is 1
void Convolution2D::backPropagation(const matrix& out_grad)
{
	//getGrad(out_grad);

	matrix temp(out_grad.row, out_grad.col);
	for (int i = 0; i < out_grad.row*out_grad.col; i++)
		temp[i] = output[i] - out_grad[i];

	update_weight(temp);
}

void Convolution2D::getGrad(matrix out_grad)
{
	matrix grad_r = out_grad.reverse();
	int padding_row = (gradient.row - 1 - filter.row + out_grad.row) / 2;
	int padding_col = (gradient.col - 1 - filter.col + out_grad.col) / 2;
	for (int i = -padding_row, x = 0; i < filter.row + padding_row - grad_r.row + 1; i++, x++)
		for (int j = -padding_col, y = 0; j < filter.col + padding_col - grad_r.col + 1; j++, y++)
		{
			gradient.getValue(x, y) = filter.Convolution(grad_r, i, j);
			if (layer_act == RELU)
				gradient.getValue(x, y) *= grad_ReLU(input.getValue(x, y));
			else if (layer_act == LRELU)
				gradient.getValue(x, y) *= grad_LReLU(input.getValue(x, y));
			else if (layer_act == SIGMOID)
				gradient.getValue(x, y) *= grad_sigmoid(input.getValue(x, y));
		}
}

void Convolution2D::update_weight(matrix out_grad)
{
	matrix dw(filter.row, filter.col);
	double db = 0.0;
	int d_size = output.row*output.col;
	for (int i = 0; i < dw.row; i++)
		for (int j = 0; j < dw.col; j++)
		{
			dw.getValue(i, j) = input.Convolution(out_grad, i, j, learning_rate);
			dw.getValue(i, j) /= static_cast<double>(d_size);
		}

	for (int i = 0; i < out_grad.row*out_grad.col; i++)
		db += out_grad[i] * learning_rate;

	db /= static_cast<double>(d_size);

	for (int i = 0; i < filter.row; i++)
		for (int j = 0; j < filter.col; j++)
			filter.getValue(i, j) -= dw.getValue(i, j);

	bias -= db;
}

void Convolution2D::apply_sigmoid()
{
	for (int i = 0; i < output.row*output.col; i++)
		sigmoid(output[i]);
}

void Convolution2D::apply_ReLU()
{
	for (int i = 0; i < output.row*output.col; i++)
		ReLU(output[i]);
}

void Convolution2D::apply_LReLU()
{
	for (int i = 0; i < output.row*output.col; i++)
		LReLU(output[i]);
}