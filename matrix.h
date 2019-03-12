#pragma once
#include "Vector.h"
#include <random>
#define RELU 1
#define LRELU 2
#define SIGMOID 3
#define SOFTMAX 4

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
	void print() const;
	void productTo(const Vector<double>& v, Vector<double>& to);
	double Convolution(const matrix& m, const int& i, const int& j, const double& a = 1.0);
	bool isValid(const int& x, const int& y);
	double& getValue(const int& i, const int& j) const;
	Vector<double> getVector(const int& _size);
	Vector<double> M2V();					// matrix to vector
	matrix productTranspose(const matrix& right);
	matrix Transpose();
	matrix applyPadding(const int& p);
	matrix reverse();

	// operator overloading
	Vector<double> operator*(const Vector<double>& right);
	double& operator[](const int& i) const;
	matrix operator*(const matrix& right);
	void operator=(const matrix& right);
	void operator+=(const double& v);
};

void v_assign_random(Vector<double>& v, const double& min, const double& max);
void v_assign_random_n(Vector<double>& v, const double& s);
double dot(const Vector<double>& left, const Vector<double>& right);
Vector<double> vsum(const Vector<double>& left, const Vector<double>& right);
matrix V2M(const Vector<double>& v, const int& i, const int& j);