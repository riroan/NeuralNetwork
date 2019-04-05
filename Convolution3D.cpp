#include"Convolution3D.h"
Convolution3D::Convolution3D(const int& x, const int& y, const int& f_w, const int& f_h, const int& c_input, const int& c_output, const int& layer_type, const char * pad, const int& stride)
	:input_channel(c_input), output_channel(c_output), input_cnt(0), pool(0), usePool(false)
{
	kernel = Vector<Vector<Convolution2D>>(output_channel);
	for (unsigned int i = 0; i < output_channel; i++)
	{
		kernel[i] = Vector<Convolution2D>(input_channel);
		for (unsigned int j = 0; j < input_channel; j++)
			kernel[i][j] = Convolution2D(y, x, f_w, f_h, layer_type, stride, pad);
	}
	output = Vector<matrix>(output_channel);
	for (unsigned int i = 0; i < output_channel; i++)
	{
		output[i] = matrix(kernel[0][0].output.row, kernel[0][0].output.col);
		output[i].assign_random(0.0, 0.0);
	}
	dx = Vector<matrix>(input_channel);
}

void Convolution3D::feedForward()
{
	assert(input_cnt == input_channel);

	for (unsigned int i = 0; i < input_channel; i++)
		for (unsigned int j = 0; j < output_channel; j++)
		{
			kernel[j][i].feedForward();
			if (usePool)
				doPooling(j, i);
			output[j] += kernel[j][i].output;
			if (usePool)
				kernel[j][i].output.resize(kernel[j][i].output.row * pool.x, kernel[j][i].output.col * pool.x);
		}
	for (unsigned int j = 0; j < output_channel; j++)
		output[j] /= input_channel;
}

void Convolution3D::setInput(const matrix& input)
{
	assert(input_cnt < input_channel);
	for (unsigned int i = 0; i < output_channel; i++)
		kernel[i][input_cnt].setInput(input);
	input_cnt++;
}

void Convolution3D::getGrad(const Vector<matrix>& out_grad)
{
	if (usePool)
		for (unsigned int i = 0; i < output_channel; i++)
			out_grad[i] = pool.getGrad(out_grad[i]);

	for (unsigned int i = 0; i < output_channel; i++)
		for (unsigned int j = 0; j < input_channel; j++)
		{
			if (usePool)
			{
				//std::cout << out_grad[i].row << ", " << out_grad[i].col << ", " << kernel[i][j].gradient.row << ", " << kernel[i][j].gradient.col << std::endl;
				kernel[i][j].getGrad(out_grad[i].elementProduct(kernel[i][j].gradient));
				//std::cout << out_grad[i].row << ", " << out_grad[i].col << ", " << kernel[i][j].gradient.row << ", " << kernel[i][j].gradient.col << std::endl;
				//out_grad[i] = kernel[i][j].gradient;
			}
			else
				kernel[i][j].getGrad(out_grad[i]);
		}

	for (unsigned int i = 0; i < input_channel; i++)
	{
		dx[i].resize(kernel[0][i].gradient.row, kernel[0][i].gradient.col);
		dx[i].assign_random(0.0, 0.0);
		for (unsigned int j = 0; j < output_channel; j++)
			dx[i] += kernel[j][i].gradient;
		dx[i] /= output_channel;
	}
}

void Convolution3D::update_weight(const Vector<matrix>& out_grad)
{
	for (unsigned int i = 0; i < output_channel; i++)
		for (unsigned int j = 0; j < input_channel; j++)
			kernel[i][j].update_weight_RMSProp(out_grad[i]);
}

Vector<double> Convolution3D::flatten()
{
	int length = output_channel * output[0].row*output[0].col;
	Vector<double> ret(length);
	int cnt = 0;
	for (unsigned int i = 0; i < output_channel; i++)
		for (int j = 0; j < output[0].row; j++)
			for (int k = 0; k < output[0].col; k++)
				ret[cnt++] = output[i].getValue(j, k);
	return ret;
}

void Convolution3D::doPooling(const int& j, const int& i)
{
	//std::cout << kernel[j][i].output << std::endl;
	kernel[j][i].output = pool.maxPool(kernel[j][i].output, kernel[j][i].gradient);
}