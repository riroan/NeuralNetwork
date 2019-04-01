#include"Convolution3D.h"

Convolution3D::Convolution3D(const int& x, const int& y, const int& f_w, const int& f_h, const int& c_input, const int& c_output, const int& layer_type, const char * pad, const int& stride)
	:input_channel(c_input), output_channel(c_output), input_cnt(0)
{
	kernel = Vector<Vector<Convolution2D>>(output_channel);
	for (int i = 0; i < output_channel; i++)
	{
		kernel[i] = Vector<Convolution2D>(input_channel);
		for (int j = 0; j < input_channel; j++)
			kernel[i][j] = Convolution2D(y, x, stride, f_w, f_h);
	}
}

//Convolution3D::Convolution3D(const int& f_w, const int& f_h, const int& c_output, const int& padding, const int& layer_type, const int& stride, const char * pad)
//{
//	kernel = Vector<Convolution2D>(c_input);
//	for (int i = 0; i < c_input; i++)
//		kernel[i] = Convolution2D(x, y, stride, f_w, f_h, padding, layer_type, pad);
//}

matrix Convolution3D::getOutput(const int& i)
{
}

void Convolution3D::feedForward()
{
	assert(input_cnt == input_channel);
	for (int i = 0; i < input_channel; i++)
		for (int j = 0; j < output_channel; j++)
			kernel[j][i].feedForward();
}

void Convolution3D::setInput(const matrix& input)
{
	assert(input_cnt < input_channel);
	for (int i = 0; i < output_channel; i++)
		kernel[i][input_cnt].setInput(input);
	input_cnt++;
}