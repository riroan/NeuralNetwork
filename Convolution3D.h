#pragma once
#include"Convolution2D.h"
#include"Pooling.h"

class Convolution3D
{
public:
	Vector<Vector<Convolution2D>> kernel;
	Vector<matrix>				  output;
	Vector<matrix>				  dx;
	unsigned					  input_channel;
	unsigned					  output_channel;
	unsigned					  input_cnt;
	Pooling						  pool;
	bool						  usePool;

	Convolution3D(const int& x, const int& y, const int& f_w, const int& f_h, const int& c_input = 1, const int& c_output = 1, const int& layer_type = RELU, const char * pad = "valid", const int& stride = 1);
	Convolution3D(const int& f_w, const int& f_h, const int& c_output, const int& padding = 0, const int& layer_type = RELU, const int& stride = 1, const char * pad = "none");
	
	void doPooling(const int& j, const int& i);
	void setInput(const matrix& input);
	void feedForward();
	void getGrad(const Vector<matrix>& out_grad);
	void update_weight(const Vector<matrix>& out_grad);
	
	Vector<double> flatten();
};