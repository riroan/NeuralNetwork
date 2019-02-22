#pragma once
#include "matrix.h"

class twoLayer
{
public:
	matrix					w[2];
	vector<double>			b[2];
	vector<vector<double>>	layers;

	twoLayer(const int& input_size, const int& hidden_size, const int& output_size);

	void ReLU(vector<double>& v);
	void sigmoid(vector<double>& v);
	void softmax(vector<double>& v);
	double loss(const vector<double>& x, const vector<double>& t);
	double CSE(const vector<double>& y, const vector<double>& t);
	vector<double> predict(const vector<double>& x);
};