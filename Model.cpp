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
		layer_A.push(new Affine(layer_C[num_of_convolution - 1]->output[0].row*layer_C[num_of_convolution - 1]->output[0].col*layer_C[num_of_convolution - 1]->output_channel, _num_output, activation));
	else
		layer_A.push(new Affine(layer_A[num_of_affine - 2]->num_output, _num_output, activation));
}

void Model::addConvolution(const int& f_w, const int& f_h, const int& output_channel, const char * pad_type, const int& activation, const int& stride, const int& padding)
{
	// Convolution layer must be in front of Affine layer.
	assert(!num_of_affine && num_of_convolution);
	num_of_layer++;
	num_of_convolution++;
	layer_C.push(new Convolution3D(layer_C[num_of_convolution - 2]->output[0].col, layer_C[num_of_convolution - 2]->output[0].row, f_w, f_h, layer_C[num_of_convolution - 2]->output_channel, output_channel, activation, pad_type, stride));
}

void Model::addConvolution(const int& _row, const int& _col, const int& f_w, const int& f_h, const int& input_channel, const int& output_channel, const int& activation, const int& stride, const int& padding, const char * pad_type)
{
	assert(!num_of_affine);
	num_of_layer++;
	num_of_convolution++;
	layer_C.push(new Convolution3D(_col, _row, f_w, f_h, input_channel, output_channel, activation, pad_type, stride));
}

void Model::setInput(const matrix& m)
{
	assert(num_of_convolution);
	layer_C[0]->setInput(m);
	//layer_C[0]->input = m;
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
			{
				for (int j = 0; j < layer_C[i]->output_channel; j++)
					layer_C[i + 1]->setInput(layer_C[i]->output[j]);
			}
		}
		layer_A[0]->input = layer_C[num_of_convolution - 1]->flatten();
	}

	for (int i = 0; i < num_of_affine; i++)
	{
		layer_A[i]->feedForward();
		if (i < num_of_affine - 1)
			layer_A[i + 1]->input = layer_A[i]->output;
	}
}

//void Model::backPropagation(const Vector<double>& y)
//{
//	assert(layer_A[num_of_affine - 1]->num_output == y.size);
//
//	layer_A[num_of_affine - 1]->getMSE(y);
//	layer_A[num_of_affine - 1]->getGrad();
//
//	for (int i = num_of_affine - 2; i >= 0; i--)
//	{
//		layer_A[i]->setGrad(layer_A[i + 1]->gradient);
//		layer_A[i]->getGrad();
//	}
//
//	if (num_of_convolution)
//	{
//		layer_C[num_of_convolution - 1]->getGrad(V2M(layer_A[0]->gradient, layer_C[num_of_convolution - 1]->output.row, layer_C[num_of_convolution - 1]->output.col));
//		for (int i = num_of_convolution - 2; i >= 0; i--)
//			layer_C[i]->getGrad(layer_C[i + 1]->gradient);
//	}
//
//	for (int i = num_of_affine - 1; i >= 0; i--)
//		layer_A[i]->update_weight_RMSProp();
//	if (num_of_convolution)
//	{
//		layer_C[num_of_convolution - 1]->update_weight(V2M(layer_A[0]->gradient, layer_C[num_of_convolution - 1]->output.row, layer_C[num_of_convolution - 1]->output.col));
//		for (int i = num_of_convolution - 2; i >= 0; i--)
//			layer_C[i]->update_weight(layer_C[i + 1]->gradient);
//	}
//
//	Error = 0;
//	if (layer_A[num_of_affine - 1]->layer_act == SIGMOID)
//		for (int i = 0; i < y.size; i++)
//			Error -= log(layer_A[num_of_affine - 1]->output[i] + 1e-8) * y[i] + log(1 - layer_A[num_of_affine - 1]->output[i] + 1e-8)*(1 - y[i]);
//	else
//		for (int i = 0; i < y.size; i++)
//			Error += (layer_A[num_of_affine - 1]->output[i] - y[i])*(layer_A[num_of_affine - 1]->output[i] - y[i]);
//}

Vector<double> Model::getOutput()
{
	return layer_A[num_of_affine - 1]->output;
}

void Model::dropout(const double& rate)
{
	for (int i = 0; i < num_of_affine - 1; i++)
		layer_A[i]->apply_dropOut(rate);
}