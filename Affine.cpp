#include"Affine.h"
#define LAMBDA 0.7

Affine::Affine(const int& _num_input, const int& _num_output, const int& activation)
	:learning_rate(0.1)
{
	layer_act = activation;
	num_input = _num_input; num_output = _num_output;
	input.resize(num_input);
	output.resize(num_output);
	out_grad.resize(num_output);
	b.resize(num_output);
	w.init_matrix(num_output, num_input);

	if (activation == RELU || activation == LRELU)
		w.assign_random_n(sqrt(num_input / 2.0));
		//w.assign_random(0.0, 0.1);
	else
		w.assign_random(0.0, 0.1);

	rms.init_matrix(num_output, num_input);
	rms.assign_random(0.0, 0.0);

	b.assign_random(0.0, 0.0);
	dropout_rate = 0.0;
	dropout = Vector<double>(num_output, 1.0);
}

void Affine::feedForward()
{
	output = w * input + b;

	if (dropout_rate > 0.0)
	{
		apply_dropOut(dropout_rate);
		output *= dropout;
		output /= dropout_rate;
	}

	if (layer_act == RELU)
		apply_ReLU(output);
	else if (layer_act == LRELU)
		apply_LReLU(output);
	else if (layer_act == SIGMOID)
		apply_sigmoid(output);
	else if (layer_act == SOFTMAX)
		apply_softmax(output);
}

void Affine::getMSE(const Vector<double>& y_t)
{
	assert(num_output == y_t.size);
	for (int i = 0; i < num_output; i++)
	{
		out_grad[i] = output[i] - y_t[i];
		if (layer_act == RELU)
			out_grad[i] *= grad_ReLU(output[i]);
		else if (layer_act == LRELU)
			out_grad[i] *= grad_LReLU(output[i]);
	}
}

void Affine::setGrad(const Vector<double>& grad)
{
	out_grad = grad;
}

void Affine::apply_sigmoid(Vector<double>& v)
{
	int v_size = v.size;
	for (int i = 0; i < v_size; i++)
		v[i] = sigmoid(v[i]);
}

void Affine::apply_ReLU(Vector<double>& v)
{
	for (int i = 0; i < v.size; i++)
		v[i] = ReLU(v[i]);
}

void Affine::apply_LReLU(Vector<double>& v)
{
	int v_size = v.size;
	for (int i = 0; i < v_size; i++)
		v[i] = LReLU(v[i]);
}

void Affine::apply_softmax(Vector<double>& v)
{
	double max = v.getMax();

	double sum = 0.0;
	for (int i = 0; i < v.size; i++)
	{
		v[i] = exp(v[i] - max);
		sum += v[i];
	}
	for (int i = 0; i < v.size; i++)
		v[i] /= sum;
}

void Affine::getGrad()
{
	if (dropout_rate > 0.0)
	{
		out_grad *= dropout;
		out_grad /= dropout_rate;
	}
	for (int i = 0; i < out_grad.size; i++)
	{
		if (layer_act == RELU)
			out_grad[i] *= grad_ReLU(output[i]);
		else if (layer_act == LRELU)
			out_grad[i] *= grad_LReLU(output[i]);
		else if (layer_act == SIGMOID)
			out_grad[i] *= grad_sigmoid(output[i]);
	}
	gradient = w.Transpose()*out_grad;
}

void Affine::setInput(const Vector<double>& _input)
{
	assert(num_input == _input.size);
	input = _input;
}

void Affine::update_weight()
{
	matrix dg = V2M(out_grad, out_grad.size, 1);
	matrix x = V2M(input, 1, input.size);
	matrix dw = dg * x;
	for (int i = 0; i < dw.row*dw.col; i++)
		w[i] -= learning_rate * dw[i];

	for (int i = 0; i < b.size; i++)
		b[i] -= dg[i] * learning_rate;
}

void Affine::update_weight_RMSProp()
{
	matrix dg = V2M(out_grad, out_grad.size, 1);
	matrix x = V2M(input, 1, input.size);
	matrix dw = dg * x;
	learning_rate = 0.001;
	const double rho = 0.9;

	rms += rms * rho + dw.elementProduct(dw) * (1 - rho);

	for (int i = 0; i < dw.row; i++)
		for (int j = 0; j < dw.col; j++)
			dw.getValue(i, j) = learning_rate / sqrt(1e-6 + rms.getValue(i, j))*dw.getValue(i, j);

	w -= dw;

	for (int i = 0; i < b.size; i++)
		b[i] -= out_grad[i] * learning_rate;
}

void Affine::apply_dropOut(const double& rate)
{
	dropout_rate = rate;
	for (int i = 0; i < dropout.size; i++)
	{
		double temp = (double)rand() / RAND_MAX;
		dropout[i] = temp < rate ? 0.0 : 1.0;
	}
}