#pragma once
#include "matrix.h"

class Network
{
public:
	int num_input;
	int num_output;
	int num_hidden;
	int num_layer;

	double alpha;
	double bias;

	vector<matrix>		   w;			// weights
	vector<int>			   num_neurons;
	vector<vector<double>> layer_grad;	// gradients
	vector<vector<double>> neurons;

	Network(const int& _num_input, const int& _num_output, const int& _num_hidden);

	void init_network(const int& _num_input, const int& _num_output, const int& _num_hidden);
	void init_network(const vector<int>& _num_neurons);

	void apply_sigmoid(vector<double>& v);
	void apply_ReLU(vector<double>& v);
	void apply_LReLU(vector<double>& v);
	void apply_softmax(vector<double>& v);

	inline double sigmoid(const double& x) { return 1.0 / (1.0 + exp(-x)); }
	inline double ReLU(const double& x) { return x > 0.0 ? x : 0.0; }
	inline double LReLU(const double& x) { return x > 0.0 ? x : 0.1*x; }

	inline double grad_sigmoid(const double& y) { return (1.0 - y)*y; }
	inline double grad_ReLU(const double& y) { return y > 0 ? 1.0 : 0.0; }
	inline double grad_LReLU(const double& y) { return y > 0 ? 1.0 : 0.1; }

	void setInput(const vector<double> v);
	void feedForward();
	void backPropagation(const vector<double>& v);
	void getGradient_MSE(const vector<double>& v);
	void update_weight();
	void printOutput();
	vector<double> getOutput();
	vector<double> gradient_product(matrix w, const vector<double>& layer);
};