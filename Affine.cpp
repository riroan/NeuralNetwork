#include"Affine.h"


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

	w.assign_random(0.0, 1.0);
	b.assign_random(0.0, 1.0);
}

void Affine::feedForward()
{
	output = w * input + b;
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
		else if (layer_act == SIGMOID)
			out_grad[i] *= grad_sigmoid(output[i]);
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
	gradient = w.Transpose()*out_grad;
	for (int i = 0; i < gradient.size; i++)
	{
		if (layer_act == RELU)
			gradient[i] *= grad_ReLU(input[i]);
		else if (layer_act == LRELU)
			gradient[i] *= grad_LReLU(input[i]);
		else if (layer_act == SIGMOID)
			gradient[i] *= grad_sigmoid(input[i]);
	}
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
		b[i] -= learning_rate * dg[i];
}