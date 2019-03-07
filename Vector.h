#pragma once
#include<iostream>
#include<cassert>

template<class T>
class Vector
{
public:
	int size;
	T * values;

	Vector()
	{
		size = 0;
		values = nullptr;
	}

	Vector(const int& _size)
	{
		size = _size;
		values = new T[size];
	}

	Vector(const int& _size, const T& assign)
	{
		size = _size;
		values = new T[size];
		for (int i = 0; i < size; i++)
			values[i] = assign;
	}

	Vector(const Vector<T>& from)
	{
		size = from.size;
		values = new T[size];
		for (int i = 0; i < size; i++)
			values[i] = from.values[i];
	}

	~Vector()
	{
		delete[] values;
	}

	void resize(const int& _size)
	{
		Vector<T> temp = *this;
		delete[] values;
		size = _size;
		values = new T[size];
		*this = temp;
	}

	void resize(const int& _size, const T& assign)
	{
		delete[] values;
		size = _size;
		values = new T[size];

		for (int i = 0; i < size; i++)
			values[i] = assign;
	}

	void print()
	{
		for (int i = 0; i < size; i++)
			printf(" %lf", values[i]);
		printf("\n");
	}

		
	T& operator[](const int& i) const
	{
		return values[i];
	}

	void operator=(const Vector<T>& v)
	{
		if (size < v.size)
			resize(v.size);
		for (int i = 0; i < v.size; i++)
			values[i] = v[i];
	}

	Vector<T> operator+(const Vector<T>& v)
	{
		assert(size == v.size);
		Vector<T> ret(size);
		for (int i = 0; i < size; i++)
			ret[i] = values[i] + v[i];
		return ret;
	}

	void assign_random(const double& min, const double& max)
	{
		for (int i = 0; i < size; i++)
			values[i] = (max - min) * ((double)rand() / RAND_MAX) + min;
	}

	double getMax()
	{
		double max = values[0];
		for (int i = 0; i < size; i++)
			if (max < values[i])
				max = values[i];
		return max;
	}

	void push(const T& item)
	{
		size++;
		resize(size);
		values[size - 1] = item;
	}

	void pop()
	{
		resize(--size);
	}
};