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

    void multiply(const Tensor& tensor, Tensor& result) {
        if (!(dim.second == tensor.dim.first
                && dim.first == result.dim.first && tensor.dim.second == result.dim.second)) {
            std::stringstream ss;
            ss<<"invalid multiplication: ("<<dim.first<<", "<<dim.second<<") * ";
            ss<<"("<<tensor.dim.first<<", "<<tensor.dim.second<<") = ";
            ss<<"("<<result.dim.first<<", "<<result.dim.second<<")";
            throw std::runtime_error(ss.str());
        }
        for (int i = 0; i < result.dim.first; i++) {
            for (int j = 0; j < result.dim.second; j++) {
                result.t[i][j] = 0.0;
                for (int k = 0; k < dim.second; k++) result.t[i][j] += t[i][k] * tensor.t[k][j];
            }
        }
    }

    double& operator[](Dim d) { return t[d.first][d.second]; }
    Tensor operator*(const Tensor& tensor) {
        Tensor result(Dim(dim.first, tensor.dim.second));
        multiply(tensor, result);
        return result;
    }

    void print(std::ostream& stream) {
        stream<<"[ ";
        for (int i = 0; i < dim.first; i++) {
            if (i > 0) stream<<"  ";
            for (int j = 0; j < dim.second; j++) stream<<t[i][j]<<" ";
            if (i == dim.first - 1) stream<<"]";
            stream<<std::endl;
        }
    }
};

}

#endif