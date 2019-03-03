#pragma once
#include"matrix.h"

class Convolution2D
{
public:
	matrix input;
	matrix filter;
	matrix output;
	bool   pad_type;
	int	   padding;
	int    stride;

	Convolution2D(const int& _stride, const int& f_w, const int& f_h, const int& _padding, const char * _pad_type = "none");
	void setInput(const matrix& _input);
	void feedForward();
};