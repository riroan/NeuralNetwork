#include"Convolution3D.h"

Convolution3D::Convolution3D(const int& x, const int& y, const int& f_w, const int& f_h, const int& c_input, const int& c_output, const int& layer_type, const char * pad, const int& stride)
	:input_channel(c_input), output_channel(c_output), input_cnt(0)
{
	kernel = Vector<Vector<Convolution2D>>(output_channel);
	for (int i = 0; i < output_channel; i++)
	{
		kernel[i] = Vector<Convolution2D>(input_channel);
		for (int j = 0; j < input_channel; j++)
			kernel[i][j] = Convolution2D(y, x, f_w, f_h, layer_type, stride);
	}
	output = Vector<matrix>(output_channel);
	for (int i = 0; i < output_channel; i++)
	{
		output[i] = matrix(kernel[0][0].output.row, kernel[0][0].output.col);
		output[i].assign_random(0.0, 0.0);
	}
	dx = Vector<matrix>(input_channel);
	for (int i = 0; i < input_channel; i++)
		dx[i].assign_random(0.0, 0.0);
}

matrix Convolution3D::getOutput(const int& i)
{
}

void Convolution3D::feedForward()
{
	assert(input_cnt == input_channel);
	for (int i = 0; i < input_channel; i++)
		for (int j = 0; j < output_channel; j++)
		{
			kernel[j][i].feedForward();
			output[j] += kernel[j][i].output;
		}
}

void Convolution3D::setInput(const matrix& input)
{
	assert(input_cnt < input_channel);
	for (int i = 0; i < output_channel; i++)
		kernel[i][input_cnt].setInput(input);
	input_cnt++;
}

void Convolution3D::getGrad(const Vector<matrix>& out_grad)
{
	for (int i = 0; i < output_channel; i++)
		for (int j = 0; j < input_channel; j++)
			kernel[i][j].getGrad(out_grad[i]);
	for (int i = 0; i < input_channel; i++)
	{
		for (int j = 0; j < output_channel; j++)
			dx[i] += kernel[j][i].gradient;
		dx[i] /= output_channel;
	}
}

void Convolution3D::updateWeight(const Vector<matrix>& out_grad)
{
	for (int i = 0; i < output_channel; i++)
		for (int j = 0; j < input_channel; j++)
			kernel[i][j].update_weight(out_grad[i]);
}

Vector<double> Convolution3D::flatten()
{
	int length = output_channel * output[0].row*output[0].col;
	Vector<double> ret(length);
	int cnt = 0;
	for (int i = 0; i < output_channel; i++)
		for (int j = 0; j < output[0].row; j++)
			for (int k = 0; k < output[0].col; k++)
				ret[cnt++] = output[i].getValue(j, k);
	return ret;
}