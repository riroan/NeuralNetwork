#include"matrix.h"
#include<iostream>
#include<future>

std::random_device rd;

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

void matrix::print() const
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			std::cout << " " << values[j + i * col];
		printf("\n");
	}
}

void matrix::init_matrix(const int& _row, const int& _col)
{
	row = _row;
	col = _col;
	values.resize(row*col);
}

void matrix::resize(const int& _row, const int& _col)
{
	row = _row;
	col = _col;
	values.resize(_row*_col);
}

void matrix::assign_random(const double& min, const double& max)
{
	values.assign_random(min, max);
}

void matrix::assign_random_n(const double& s)
{
	values.assign_random_n(s);
}

void matrix::productTo(const Vector<double>& v, Vector<double>& to)
{
	for (int i = 0; i < row; i++)
	{
		to[i] = 0.0;
		for (int j = 0; j < col; j++)
			to[i] += getValue(i, j)*v[j];
	}
}

matrix matrix::elementProduct(const matrix& right)
{
	assert(row == right.row);
	assert(col == right.col);
	matrix ret(row, col);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			ret.getValue(i, j) = getValue(i, j) * right.getValue(i, j);
	return ret;
}

Vector<double> matrix::operator*(const Vector<double>& right)
{
	assert(col == right.size);
	Vector<double> ret(row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			ret[i] += getValue(i, j)*right.values[j];

	return ret;
}

matrix matrix::operator+(const matrix& right)
{
	assert(row == right.row);
	assert(col == right.col);
	matrix ret(row, col);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			ret.getValue(i, j) = getValue(i, j) + right.getValue(i, j);
	return ret;
}

double& matrix::getValue(const int& i, const int& j) const
{
	return values[i*col + j];
}

matrix matrix::operator*(const matrix& right)
{
	assert(col == right.row);
	matrix ret(row, right.col);
	ret.assign_random(0.0, 0.0);

	for (int i = 0; i < ret.row; i++)
		for (int j = 0; j < ret.col; j++)
			for (int k = 0; k < col; k++)
				ret.getValue(i, j) += getValue(i, k)*right.getValue(k, j);

	return ret;
}

double& matrix::operator[](const int& i) const
{
	return values[i];
}

void matrix::operator=(const matrix& right)
{
	row = right.row;
	col = right.col;
	values.resize(row * col);
	for (int i = 0; i < row * col; i++)
		values[i] = right[i];
}

matrix matrix::productTranspose(const matrix& right)
{
	matrix temp(right.col, right.row);
	for (int i = 0; i < right.row; i++)
		for (int j = 0; j < right.col; j++)
			temp.getValue(j, i) = right.getValue(i, j);
	return (*this)*temp;
}

matrix matrix::Transpose()
{
	matrix temp(col, row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			temp.getValue(j, i) = getValue(i, j);
	return temp;
}

Vector<double> matrix::M2V()
{
	Vector<double> ret;
	ret = values;
	ret.size = row * col;
	return ret;
}

matrix matrix::applyPadding(const int& p)
{
	matrix ret(row + 2 * p, col + 2 * p);
	ret.assign_random(0.0, 0.0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			ret.getValue(i + p, j + p) = getValue(i, j);
	return ret;
}

bool matrix::isValid(const int& i, const int& j)
{
	return i >= 0 && i < row&&j >= 0 && j < col;
}

double matrix::Convolution(const matrix& m, const int& x, const int& y, const double& a)
{
	double sum = 0.0;
	for (int i = 0; i < m.row; i++)
		for (int j = 0; j < m.col; j++)
			if (isValid(x + i, y + j))
				sum += a * getValue(x + i, y + j) * m.getValue(i, j);
	return sum;
}

void matrix::operator+=(const double& v)
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			getValue(i, j) += v;
}

void matrix::operator/=(const double& v)
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			getValue(i, j) /= v;
}

matrix matrix::reverse()
{
	matrix ret(row, col);
	for (int i = 0; i < row*col; i++)
		ret[i] = values[row*col - 1 - i];
	return ret;
}

matrix V2M(const Vector<double>& v, const int& i, const int& j)
{
	//assert(v.size == i * j);
	matrix ret(i, j);
	ret.values = v;
	return ret;
}

void matrix::operator-=(const matrix& right)
{
	assert(row == right.row);
	assert(col == right.col);
	for (int i = 0; i < row*col; i++)
		values[i] -= right[i];
}

void matrix::operator+=(const matrix& right)
{
	assert(row == right.row);
	assert(col == right.col);
	for (int i = 0; i < row*col; i++)
		values[i] += right[i];
}

matrix matrix::operator*(const double& v)
{
	matrix ret(row, col);
	for (int i = 0; i < row*col; i++)
		ret[i] = values[i] * v;
	return ret;
}

matrix matrix::operator-(const matrix& right)
{
	assert(row == right.row);
	assert(col == right.col);
	matrix ret(row, col);
	for (int i = 0; i < row*col; i++)
		ret[i] = values[i] - right[i];
	return ret;
}