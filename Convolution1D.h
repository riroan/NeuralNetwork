#pragma once
#include"Vector.h"
#include<random>
#include<cstring>
using namespace std;

class Convolution1D
{
public:
	Vector<double> input;
	Vector<double> output;
	Vector<double> filter;
	bool		   pad_type;
	int			   padding;
	int			   stride;
	int			   f_size;
	int			   n_channel;

	Convolution1D(const int& _stride, const int& _f_size, const int& _padding, const char * _pad_type = "none");
	void setInput(const Vector<double>& _input);
	void feedForward();
	void backPropagation(const Vector<double>& grad);
};