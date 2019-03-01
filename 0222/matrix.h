#pragma once
#include <vector>
#include <random>
#include <cassert>
using namespace std;

class matrix
{
public:
	// member variable
	int row;
	int col;
	vector<vector<double>> values;

	// constructor
	matrix();
	matrix(const int& _row, const int& _col);

	// member function
	void init_matrix(const int& _row, const int& _col);
	void resize(const int& _row, const int& _col);
	void assign_random(const double& min, const double& max);
	void assign_random_n();
	void print();
	vector<double> getVector(const int& _size);
	matrix productTranspose(const matrix& right);
	matrix Transpose();

	// operator overloading
	vector<double> operator*(const vector<double>& right);
	vector<double>& operator[](const int& i);
	matrix operator*(const matrix& right);
};

void v_assign_random(vector<double>& v, const double& min, const double& max);
void v_assign_random_n(vector<double>& v);
double dot(const vector<double>& left, const vector<double>& right);
vector<double> vsum(const vector<double>& left, const vector<double>& right);