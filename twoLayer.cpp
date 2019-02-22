#include"twoLayer.h"

twoLayer::twoLayer(const int& input_size, const int& hidden_size, const int& output_size)
{
	w[0].init_matrix(input_size, hidden_size);
	w[1].init_matrix(hidden_size, output_size);
	w[0].assign_random_n();
	w[1].assign_random_n();

	b[0].resize(hidden_size);
	b[1].resize(output_size);

	layers.resize(2);
}

void twoLayer::sigmoid(vector<double>& v)
{
	int v_size = v.size();
	for (int i = 0; i < v_size; i++)
		v[i] = 1 / (1 + exp(-v[i]));
}

void twoLayer::ReLU(vector<double>& v)
{
	int v_size = v.size();
	for (int i = 0; i < v_size; i++)
		v[i] = v[i] > 0.0 ? v[i] : 0.0;
}

void twoLayer::softmax(vector<double>& v)
{
	int v_size = v.size();
	double max = 0.0;
	for (auto i : v)
		if (max < i)
			max = i;
	double sum = 0.0;
	for (auto i : v)
	{
		i -= max;
		i = exp(i);
		sum += i;
	}
	for (auto i : v)
		i /= sum;
}

vector<double> twoLayer::predict(const vector<double>& x)
{
	vector<double> a1 = w[0] * x;
	a1 = vsum(a1, b[0]);
	sigmoid(a1);

	vector<double> a2 = w[1] * a1;
	a2 = vsum(a2, b[1]);
	softmax(a2);

	return a2;
}

double twoLayer::loss(const vector<double>& x, const vector<double>& t)
{
	vector<double> y = predict(x);

	return CSE(y, t);
}

double twoLayer::CSE(const vector<double>& y, const vector<double>& t)
{
	assert(y.size() == t.size());
	double sum = 0.0;
	int v_size = y.size();

	for (int i=0;i<v_size;i++)
		sum -= (t[i] * log(y[i] + 1e-7));

	return sum;
}