#include"Convolution2D.h"

Convolution2D::Convolution2D(const int& _stride, const int& f_w, const int& f_h, const int& _padding, const int& _layer_act, const char * _pad_type)
	:padding(_padding), stride(_stride), filter(f_h, f_w), learning_rate(0.15)
{
	layer_act = _layer_act;

	filter.assign_random(0.0, 1.0);

	if (strcmp(_pad_type, "same") == 0)
		pad_type = true;
	else
		pad_type = false;

	bias = (double)rand() / RAND_MAX;
}

void Convolution2D::setInput(const matrix& _input)
{
	input = _input;
	gradient.init_matrix(input.row, input.col);
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
		{
			output.getValue(x, y) = input.Convolution(filter, i, j);
			output.getValue(x, y) += bias;
		}
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
	getGrad(out_grad);
	update_weight(out_grad);
}

void Convolution2D::getGrad(matrix out_grad)
{
	matrix grad_r = out_grad.reverse();
	for (int i = -1, x = 0; i < filter.row + 1 - grad_r.row + 1; i++, x++)
		for (int j = -1, y = 0; j < filter.col + 1 - grad_r.col + 1; j++, y++)
			gradient.getValue(x, y) = filter.Convolution(grad_r, i, j);
}

void Convolution2D::update_weight(matrix out_grad)
{
	matrix dw(filter.row, filter.col);
	for (int i = 0; i < dw.row; i++)
		for (int j = 0; j < dw.col; j++)
		{
			dw.getValue(i, j) = input.Convolution(out_grad, i, j);
			if (layer_act == RELU)
				dw.getValue(i, j) *= grad_ReLU(output.getValue(i, j));
			else if (layer_act == LRELU)
				dw.getValue(i, j) *= grad_LReLU(output.getValue(i, j));
			else if (layer_act == SIGMOID)
				dw.getValue(i, j) *= grad_sigmoid(output.getValue(i, j));
		}

	for (int i = 0; i < filter.row; i++)
		for (int j = 0; j < filter.col; j++)
		{
			filter.getValue(i, j) -= learning_rate * dw.getValue(i, j);
			bias -= learning_rate * out_grad.getValue(i, j);
		}
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