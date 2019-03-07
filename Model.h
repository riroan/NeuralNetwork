#pragma once
#include"Affine.h"
#include"Convolution2D.h"

class Model
{
public:
	Vector<Affine *>		  layer_A;
	Vector<Convolution2D *>   layer_C;
	int						  num_of_layer;
	int 					  num_of_affine;
	int						  num_of_convolution;
	double					  Error;

	Model();
	~Model();

	void addAffine(const int& _num_input, const int& _num_output, const int& activation);
	void addAffine(const int& _num_output, const int& activation = RELU);
	void addConvolution(const int& f_w, const int& f_h, const int& activation = RELU, const int& stride = 1, const int& padding = 0, const char * pad_type = "none");

	void setInput(const matrix& m);
	void setInput(const Vector<double>& v);
	void forwardPropagation();
	void backPropagation(const Vector<double>& y);

	Vector<double> getOutput();
};