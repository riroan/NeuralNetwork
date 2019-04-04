#pragma once
#include"Affine.h"
#include"Convolution3D.h"

class Model
{
public:
	Vector<Affine *>		  layer_A;
	Vector<Convolution3D *>   layer_C;
	short					  num_of_layer;
	short					  num_of_affine;
	short					  num_of_convolution;
	double					  Error;

	Model();
	~Model();

	void addAffine(const int& _num_input, const int& _num_output, const int& activation);
	void addAffine(const int& _num_output, const int& activation = RELU);
	void addConvolution(const int& f_w, const int& f_h, const int& output_channel = 1, const char * pad_type = "valid", const int& activation = RELU, const int& stride = 1, const int& padding = 0);
	void addConvolution(const int& _row, const int& _col, const int& f_w, const int& f_h, const int& input_channel = 1, const int& output_channel = 1, const int& activation = RELU, const int& stride = 1, const char * pad_type = "none");
	void maxPool(const int& size);

	void dropout(const double& rate = 0.5);

	void setInput(const matrix& m);
	void setInput(const Vector<double>& v);
	void forwardPropagation();
	void backPropagation(const Vector<double>& y);

	Vector<double> getOutput();
};