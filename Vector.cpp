#include"Vector.h"

std::ostream &operator<<(std::ostream &os, const Vector<double>& v)
{
	os << "[ ";
	for (int i = 0; i < v.size; i++)
		os << " " << v[i];
	os << "]\n";
	return os;
}