#pragma once
#include "Vector.h"
#define IDENTITY 0
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
	Vector<double> M2V();								// matrix to vector
	matrix productTranspose(const matrix& right);
	matrix Transpose();
	matrix applyPadding(const int& p);
	matrix reverse() const;
	matrix elementProduct(const matrix& right);

	// operator overloading
	Vector<double> operator*(const Vector<double>& right);
	double& operator[](const int& i) const;
	matrix operator+(const matrix& right);
	matrix operator*(const matrix& right);
	matrix operator*(const double& v);
	matrix operator-(const matrix& right);
	void operator=(const matrix& right);
	void operator+=(const double& v);
	void operator/=(const double& v);
	void operator-=(const matrix& right);
	void operator+=(const matrix& right);

	friend std::ostream &operator<<(std::ostream &os, const matrix& v);
};

matrix V2M(const Vector<double>& v, const int& i, const int& j);
Vector<matrix> V2VM(const Vector<double>& v, const int& len, const int& row, const int& col);