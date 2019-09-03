#ifndef ADER_MATH_HPP
#define ADER_MATH_HPP

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>
#include <valarray>
#include <numeric>

namespace ader {

// 今は行列のみ対応
using Dim = std::pair<int, int>;

class Tensor {
    Dim dim;
    std::valarray<std::valarray<double>> t;

public:
    Tensor(Dim d) : dim(d), t(std::valarray<double>(d.second), d.first) {}

    Dim getDim() const { return dim; }

    void add(const Tensor& tensor, Tensor& result) const;
    void multiply(const Tensor& tensor, Tensor& result) const;

    Tensor& operator=(const Tensor& tensor);
    double& operator()(int i, int j) { return t[i][j]; }
    Tensor operator+(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;

    void print(std::ostream& stream) const;
    std::string typeAsStr() const;
};

}

#endif