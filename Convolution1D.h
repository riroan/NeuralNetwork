#pragma once
#include<vector>
#include<random>
#include<cstring>
#include<algorithm>
using namespace std;

class Convolution1D
{
public:
	vector<double> input;
	vector<double> output;
	vector<double> filter;
	bool		   pad_type;
	int			   padding;
	int			   stride;
	int			   f_size;
	int			   n_channel;

	Convolution1D(const int& _stride, const int& _f_size, const int& _padding, const char * _pad_type = "none");
	void setInput(const vector<double>& _input);
	void feedForward();
	void backPropagation(const vector<double>& grad);
};