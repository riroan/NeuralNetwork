#include"matrix.h"
#include<iostream>

random_device rd;

matrix::matrix(const int& _row, const int& _col)
	:row(_row), col(_col)
{
	init_matrix(_row, _col);
}

matrix::matrix()
{
	row = 0;
	col = 0;
	values.resize(0);
}

void matrix::print()
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			printf(" %.2lf", values[i][j]);
		printf("\n");
	}
}

void matrix::init_matrix(const int& _row, const int& _col)
{
	row = _row;
	col = _col;
	values.resize(row);
	for (int i = 0; i < row; i++)
		values[i].resize(col, 0.0);
}

void matrix::resize(const int& _row, const int& _col)
{
	row = _row;
	col = _col;
	for (int i = 0; i < row; i++)
		values[i].resize(col, 0.0);
	values.resize(row, getVector(col));
}

void matrix::assign_random(const double& min, const double& max)
{
	for (int i = 0; i < row; i++)
		v_assign_random(values[i], min, max);
}

void matrix::assign_random_n()
{
	for (int i = 0; i < row; i++)
		v_assign_random_n(values[i]);
}

vector<double> matrix::getVector(const int& _size)
{
	vector<double> ret(_size);
	for (int i = 0; i < _size; i++)
		ret[i] = 0.0;
	return ret;
}

vector<double> matrix::operator*(const vector<double>& right)
{
	assert(col == right.size());
	vector<double> ret(row);
	for (int i = 0; i < row; i++)
		ret[i] = dot(values[i], right);
	return ret;
}

matrix matrix::operator*(const matrix& right)
{
	assert(col == right.col);
	matrix ret(row, right.row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < right.row; j++)
			ret[i][j] = dot(values[i], right.values[j]);
	return ret;
}

vector<double>& matrix::operator[](const int& i)
{
	return values[i];
}

matrix matrix::productTranspose(const matrix& right)
{
	matrix temp(right.col, right.row);
	for (int i = 0; i < right.row; i++)
		for (int j = 0; j < right.col; j++)
			temp[j][i] = right.values[i][j];
	return (*this)*temp;
}

matrix matrix::Transpose()
{
	matrix temp(col, row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			temp[j][i] = values[i][j];
	return temp;
}

double dot(const vector<double>& left, const vector<double>& right)
{
	assert(left.size() == right.size());
	double ret = 0.0;

	for (unsigned int i = 0; i < left.size(); i++)
		ret += left[i] * right[i];
	return ret;
}

vector<double> vsum(const vector<double>& left, const vector<double>& right)
{
	assert(left.size() == right.size());
	vector<double> ret(left.size());

	int l_size = left.size();

	for (unsigned int i = 0; i < l_size; i++)
		ret[i] = left[i] + right[i];
	return ret;
}

void v_assign_random(vector<double>& v, const double& min, const double& max)
{
	uniform_real_distribution<double> distribution(min, max);
	int v_size = v.size();

	for (int i = 0; i < v_size; i++)
		v[i] = distribution(rd);
}

void v_assign_random_n(vector<double>& v)
{
	normal_distribution<double> distribution(0.0, 1.0);
	int v_size = v.size();

	for (int i = 0; i < v_size; i++)
		v[i] = distribution(rd);
}