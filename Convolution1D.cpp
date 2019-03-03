#include"Convolution1D.h"
#include<iostream>
using namespace std;

Convolution1D::Convolution1D(const int& _stride, const int& _f_size, const int& _padding, const char * _pad_type)
	:padding(_padding), stride(_stride), f_size(_f_size)
{
	random_device rd;
	uniform_real_distribution distribution(0.0, 1.0);
	filter.resize(f_size);
	for (int i = 0; i < filter.size; i++)
		filter[i] = distribution(rd);
	if (strcmp(_pad_type, "same") == 0)
		pad_type = true;
	else
		pad_type = false;
}

void Convolution1D::setInput(const Vector<double>& _input)
{
	input = _input;
	if (pad_type)
	{
		padding = (stride*(input.size - 1) + f_size - input.size) / 2;
		output.resize(input.size, 0.0);
	}
}

void Convolution1D::feedForward()
{
	int current_pos = -padding;
	for (int i = 0; i < output.size; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < f_size; j++)
			if (current_pos >= 0 && current_pos + j < input.size)
				sum += filter[j] * input[current_pos + j];
		output[i] = sum;
		current_pos += stride;
	}
}

void Convolution1D::backPropagation(const Vector<double>& grad)
{
	Vector<double> get_grad = filter;

}