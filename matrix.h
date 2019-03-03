#pragma once
#include "Vector.h"
#include <random>
#include <cassert>

class matrix
{
public:
	// member variable
	int row;
	int col;
	Vector<double> values;

	// constructor
	matrix();
	matrix(const int& _row, const int& _col);

	// member function
	void init_matrix(const int& _row, const int& _col);
	void resize(const int& _row, const int& _col);
	void assign_random(const double& min, const double& max);
	void assign_random_n(const double& s);
	void print();
	void productTo(const Vector<double>& v, Vector<double>& to);
	double Convolution(matrix m, const int& i, const int& j);
	bool isValid(const int& x, const int& y);
	double& getValue(const int& i, const int& j);
	Vector<double> getVector(const int& _size);
	Vector<double> M2V();					// matrix to vector
	matrix productTranspose(matrix right);
	matrix Transpose();

	// operator overloading
	Vector<double> operator*(const Vector<double>& right);
	double& operator[](const int& i);
	matrix operator*(matrix right);
	void operator=(matrix right);
};

void v_assign_random(Vector<double>& v, const double& min, const double& max);
void v_assign_random_n(Vector<double>& v, const double& s);
double dot(const Vector<double>& left, const Vector<double>& right);
Vector<double> vsum(const Vector<double>& left, const Vector<double>& right);