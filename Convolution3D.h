#pragma once
#include"Convolution2D.h"

class Convolution3D
{
public:
	Vector<Vector<Convolution2D>> kernel;
	Vector<matrix>				  output;
	unsigned					  input_channel;
	unsigned					  output_channel;
	unsigned					  input_cnt;

	Convolution3D(const int& x, const int& y, const int& f_w, const int& f_h, const int& c_input = 1, const int& c_output = 1, const int& layer_type = RELU, const char * pad = "valid", const int& stride = 1);
	Convolution3D(const int& f_w, const int& f_h, const int& c_output, const int& padding = 0, const int& layer_type = RELU, const int& stride = 1, const char * pad = "none");
	
	matrix getOutput(const int& i);
	void setInput(const matrix& input);
	void feedForward();
	void backPropagation();
	
};