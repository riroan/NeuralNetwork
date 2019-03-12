#include "Model.h"
#include<thread>
#include<future>
using namespace std;

Model::Model()
	:layer_A(0), layer_C(0), num_of_layer(0), num_of_affine(0), num_of_convolution(0), Error(0.0)
{}

Model::~Model()
{
	for (int i = 0; i < num_of_affine; i++)
		delete layer_A[i];
	for (int i = 0; i < num_of_convolution; i++)
		delete layer_C[i];
}

void Model::addAffine(const int& _num_input, const int& _num_output, const int& activation)
{
	num_of_layer++;
	num_of_affine++;
	layer_A.push(new Affine(_num_input, _num_output, activation));
}

void Model::addAffine(const int& _num_output, const int& activation)
{
	// if you want to omit num_input, this layer must not be first.
	assert(num_of_layer);
	num_of_layer++;
	num_of_affine++;
	if (num_of_convolution&&num_of_affine == 1)
		layer_A.push(new Affine(layer_C[num_of_convolution - 1]->output.row*layer_C[num_of_convolution - 1]->output.col, _num_output, activation));
	else
		layer_A.push(new Affine(layer_A[num_of_layer - 2]->num_output, _num_output, activation));
}

void Model::addConvolution(const int& f_w, const int& f_h, const int& activation, const int& stride, const int& padding, const char * pad_type)
{
	// Convolution layer must be in front of Affine layer.
	assert(!num_of_affine && num_of_convolution);
	num_of_layer++;
	num_of_convolution++;
	layer_C.push(new Convolution2D(layer_C[num_of_convolution - 2]->output.row, layer_C[num_of_convolution - 2]->output.col, stride, f_w, f_h, padding, activation, pad_type));
}

void Model::addConvolution(const int& _row, const int& _col, const int& f_w, const int& f_h, const int& activation, const int& stride, const int& padding, const char * pad_type)
{
	assert(!num_of_affine);
	num_of_layer++;
	num_of_convolution++;
	layer_C.push(new Convolution2D(_row, _col, stride, f_w, f_h, padding, activation, pad_type));
}

void Model::setInput(const matrix& m)
{
	assert(num_of_convolution);
	layer_C[0]->input = m;
}

void Model::setInput(const Vector<double>& v)
{
	assert(num_of_affine);
	layer_A[0]->input = v;
}

void Model::forwardPropagation()
{
	if (num_of_convolution)
	{
		for (int i = 0; i < num_of_convolution; i++)
		{
			layer_C[i]->feedForward();
			if (i < num_of_convolution - 1)
				layer_C[i + 1]->input = layer_C[i]->output;
		}
		layer_A[0]->input = layer_C[num_of_convolution - 1]->output.M2V();
	}

	for (int i = 0; i < num_of_affine; i++)
	{
			layer_A[i]->feedForward();
			if (i < num_of_affine - 1)
				layer_A[i + 1]->input = layer_A[i]->output;
	}
}

void Model::backPropagation(const Vector<double>& y)
{
	assert(layer_A[num_of_affine - 1]->num_output == y.size);
	layer_A[num_of_affine - 1]->getMSE(y);
	layer_A[num_of_affine - 1]->getGrad();

	for (int i = num_of_affine - 2; i >= 0; i--)
	{
		layer_A[i]->setGrad(layer_A[i + 1]->gradient);
		layer_A[i]->getGrad();
	}

	if (num_of_convolution)
	{
		layer_C[num_of_convolution - 1]->getGrad(V2M(layer_A[0]->gradient, layer_C[num_of_convolution - 1]->output.row, layer_C[num_of_convolution - 1]->output.col));
		for (int i = num_of_convolution - 2; i >= 0; i--)
			layer_C[i]->getGrad(layer_C[i + 1]->gradient);
	}

	for (int i = num_of_affine - 1; i >= 0; i--)
		layer_A[i]->update_weight();
	if (num_of_convolution)
	{
		layer_C[num_of_convolution - 1]->update_weight(V2M(layer_A[0]->gradient, layer_C[num_of_convolution - 1]->output.row, layer_C[num_of_convolution - 1]->output.col));
		for (int i = num_of_convolution - 2; i >= 0; i--)
			layer_C[i]->update_weight(layer_C[i + 1]->gradient);
	}

	Error = 0;
	for (int i = 0; i < y.size; i++)
		Error += pow((layer_A[num_of_affine - 1]->output[i] - y[i]), 2.0);
}

Vector<double> Model::getOutput()
{
	return layer_A[num_of_affine - 1]->output;
}