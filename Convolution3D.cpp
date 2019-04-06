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

	// no thread : 3.5
	// 2 threads : 3.0 ~ 3.5
	// 4 threads : 2.9~3.1

	auto f = [&](const int& i_start, const int& i_end, const int& j_start, const int& j_end)
	{
		for (unsigned int i = i_start; i < i_end; i++)
			for (unsigned int j = j_start; j < j_end; j++)
			{
				kernel[j][i].feedForward();
				if (usePool)
					doPooling(j, i);
				output[j] += kernel[j][i].output;
				if (usePool)
					kernel[j][i].output.resize(kernel[j][i].output.row * pool.x, kernel[j][i].output.col * pool.x);
			}
	};

	Vector<std::future<void>> futures(4);
	futures[0] = std::async(f, 0, input_channel / 2, 0, output_channel/2);
	futures[1] = std::async(f, input_channel / 2, input_channel, 0, output_channel / 2);
	futures[2] = std::async(f, 0, input_channel / 2, output_channel / 2, output_channel);
	futures[3] = std::async(f, input_channel / 2, input_channel, output_channel / 2, output_channel);
	
	for (int i = 0; i < 4; i++)
		futures[i].get();

	// 0.0002 0.0007 0.0002

	//for (unsigned int i = 0; i < input_channel; i++)
	//	for (unsigned int j = 0; j < output_channel; j++)
	//	{
	//		kernel[j][i].feedForward();
	//		if (usePool)
	//			doPooling(j, i);
	//		output[j] += kernel[j][i].output;
	//		if (usePool)
	//			kernel[j][i].output.resize(kernel[j][i].output.row * pool.x, kernel[j][i].output.col * pool.x);
	//	}

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

	// no thread : 0.4,0.3,0.2

	auto f = [&](const int& i_s, const int& i_e, const int& j_s, const int& j_e)
	{
		for (unsigned int i = i_s; i < i_e; i++)
			for (unsigned int j = j_s; j < j_e; j++)
				if (usePool)
					kernel[i][j].getGrad(out_grad[i].elementProduct(kernel[i][j].gradient));
				else
					kernel[i][j].getGrad(out_grad[i]);
	};
	Vector<std::future<void>> futures(4);
	futures[0] = std::async(f, 0, output_channel / 2, 0, input_channel / 2);
	futures[1] = std::async(f, 0, output_channel / 2, input_channel / 2, input_channel);
	futures[2] = std::async(f, output_channel / 2, output_channel, 0, input_channel / 2);
	futures[3] = std::async(f, output_channel / 2, output_channel, input_channel / 2, input_channel);
	for (int i = 0; i < 4; i++)
		futures[i].get();

	//for (unsigned int i = 0; i < output_channel; i++)
	//	for (unsigned int j = 0; j < input_channel; j++)
	//		if (usePool)
	//			kernel[i][j].getGrad(out_grad[i].elementProduct(kernel[i][j].gradient));
	//		else
	//			kernel[i][j].getGrad(out_grad[i]);

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
			kernel[i][j].update_weight(out_grad[i]);
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