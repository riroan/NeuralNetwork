#include "Network.h"
#include<iostream>

Network::Network(const int& _num_input, const int& _num_output, const int& _num_hidden)
	:w(_num_hidden+1)
{
	init_network(_num_input, _num_output, _num_hidden);
}

void Network::init_network(const vector<int>& _num_neurons)
{
	neurons.resize(num_layer);
	layer_grad.resize(num_layer);
	for (int i = 0; i < num_layer; i++)
	{
		neurons[i].resize(_num_neurons[i], 1.0);
		neurons[i][_num_neurons[i] - 1] = bias;
		layer_grad[i].resize(_num_neurons[i], 1.0);
	}

	w.resize(num_layer - 1);
	for (int i = 0; i < num_layer - 1; i++)
	{
		w[i].init_matrix(_num_neurons[i + 1] - 1, _num_neurons[i]);
		w[i].assign_random(0.0, 1.0);
		//w[i].print();
	}
}

void Network::init_network(const int& _num_input, const int& _num_output, const int& _num_hidden)
{
	num_input = _num_input;
	num_output = _num_output;
	num_hidden = _num_hidden;
	num_layer = num_hidden + 2;

	num_neurons.resize(num_layer);

	for (int i = 0; i < num_layer - 1; i++)	// except output layer
		num_neurons[i] = num_input + 1;
	num_neurons[num_layer - 1] = num_output + 1;
	
	bias = 1;
	alpha = 0.1;

	init_network(num_neurons);
}

void Network::apply_sigmoid(vector<double>& v)
{
	int v_size = v.size();
	for (int i = 0; i < v_size - 1; i++)
		v[i] = sigmoid(v[i]);
}

void Network::apply_ReLU(vector<double>& v)
{
	int v_size = v.size();
	for (int i = 0; i < v_size - 1; i++)
		v[i] = ReLU(v[i]);
}

void Network::apply_LReLU(vector<double>& v)
{
	int v_size = v.size();
	for (int i = 0; i < v_size - 1; i++)
		v[i] = LReLU(v[i]);
}

void Network::apply_softmax(vector<double>& v)
{
	double max = v[0];

	int v_size = v.size();
	for (int i = 1; i < v_size - 1; i++)
		if (v[i] > max)
			max = v[i];

	double sum = 0.0;
	for (int i = 0; i < v_size - 1; i++)
	{
		v[i] = exp(v[i] - max);
		sum += v[i];
	}
	for (int i = 0; i < v_size - 1; i++)
		v[i] /= sum;
}

void Network::feedForward()
{
	int w_size = w.size();
	for (int i = 0; i < w_size; i++)
	{
		vector<double> temp = w[i] * neurons[i];
		int t_size = temp.size();
		for (int j = 0; j < t_size; j++)
			neurons[i + 1][j] = temp[j];
		apply_ReLU(neurons[i + 1]);
	}
}

vector<double> Network::getOutput()
{
	vector<double> ret(num_output);
	for (int i = 0; i < num_output; i++)
		ret[i] = neurons[num_layer - 1][i];
	return ret;
}

void Network::setInput(const vector<double> v)
{
	int v_size = v.size();
	assert(num_input == v_size);
	for (int i = 0; i < num_input; i++)
		neurons[0][i] = v[i];
}

void Network::backPropagation(const vector<double>& v)
{
	getGradient_MSE(v);
	update_weight();
}

void printv(const vector<double>& v)
{
	for (auto i : v)
		cout << " " << i;
	cout << endl;
}

void Network::update_weight()
{
	int w_size = w.size();
	for (int r = w_size - 1; r >= 0; r--)
		for (int i = 0; i < w[r].row; i++)
			for (int j = 0; j < w[r].col; j++)
			{
				double delta = alpha * layer_grad[r + 1][i] * neurons[r][j];
				w[r][i][j] += delta;
			}
}

void Network::getGradient_MSE(const vector<double>& v)
{
	int last = num_layer - 1;
	int l_size = layer_grad[last].size();
	for (int i = 0; i < l_size - 1; i++)
	{
		double last_value = neurons[last][i];
		// MSE
		layer_grad[last][i] = (v[i] - last_value)*grad_ReLU(last_value);
	}

	int w_size = w.size();
	for (int i = w_size - 1; i >= 0; i--)
	{
		vector<double> temp = gradient_product(w[i].Transpose(), layer_grad[i + 1]);
		int t_size = temp.size();
		for (int j = 0; j < t_size; j++)
			layer_grad[i][j] = temp[j];

		int n_size = neurons[i].size();
		for (int j = 0; j < n_size - 1; j++)
			layer_grad[i][j] *= grad_ReLU(neurons[i][j]);
	}
}

vector<double> Network::gradient_product(matrix w, const vector<double>& layer)
{
	assert(w.col == layer.size() - 1);
	int l_size = layer.size() - 1;
	vector<double> temp(l_size);
	for (int i = 0; i < l_size; i++)
		temp[i] = layer[i];

	return w * temp;
}

void Network::printOutput()
{
	for (int i = 0; i < num_output; i++)
		cout << neurons[num_layer - 1][i] << " ";
	cout << endl;
}