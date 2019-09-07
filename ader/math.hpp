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
    Tensor(const std::valarray<std::valarray<double>>& tt) : t(tt) {
        if (t.size() == 0) dim = Dim(0, 0);
        else dim = Dim(t.size(), t[0].size());
    }
    Tensor(std::valarray<std::valarray<double>>&& tt) : t(std::move(tt)) {
        if (t.size() == 0) dim = Dim(0, 0);
        else dim = Dim(t.size(), t[0].size());
    }
    Tensor(Dim d, double init = 0.0) : dim(d), t(std::valarray<double>(init, d.second), d.first) {}
    Tensor(const Tensor& tensor) : dim(tensor.dim), t(tensor.t) {}

    Dim getDim() const { return dim; }

    void clear();
    Tensor transpose() const;
    Tensor sin() const;
    Tensor cos() const;
    double sum() const;
    void add(const Tensor& tensor, Tensor& result) const;
    void sub(const Tensor& tensor, Tensor& result) const;
    void multiply(const Tensor& tensor, Tensor& result) const;
    void hadamard(const Tensor& tensor, Tensor& result) const;
    Tensor hadamard(const Tensor& tensor) const;

    Tensor& operator=(const Tensor& tensor);
    double& operator()(int i, int j) { return t[i][j]; }
    Tensor operator+(const Tensor& tensor) const;
    Tensor operator-(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;

    friend Tensor operator-(const Tensor& tensor);
    friend Tensor operator*(double d, const Tensor& tensor);
    friend std::ostream& operator<<(std::ostream&, const Tensor&);

    void print(std::ostream& stream) const;
    std::string typeAsStr() const;
};

}

#endif