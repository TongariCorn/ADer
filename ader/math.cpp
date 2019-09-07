#include "math.hpp"

namespace ader {

void Tensor::clear() {
    for (int i = 0; i < dim.first; i++) t[i] = 0.0;
}

Tensor Tensor::transpose() const {
    auto tt = Tensor(Dim(dim.second, dim.first));
    for (int i = 0; i < dim.second; i++)
        for (int j = 0; j < dim.first; j++) tt(i,j) = t[j][i];
    return tt;
}

void Tensor::add(const Tensor& tensor, Tensor& result) const {
    if (!(dim == tensor.dim && dim == result.dim)) {
        std::stringstream ss;
        ss<<"invalid addition: "<<typeAsStr()<<" + "<<tensor.typeAsStr()<<" = "<<result.typeAsStr();
        throw std::runtime_error(ss.str());
    }
    for (int i = 0; i < result.dim.first; i++) result.t[i] = t[i] + tensor.t[i];
}

void Tensor::multiply(const Tensor& tensor, Tensor& result) const {
    if (!(dim.second == tensor.dim.first
            && dim.first == result.dim.first && tensor.dim.second == result.dim.second)) {
        std::stringstream ss;
        ss<<"invalid multiplication: "<<typeAsStr()<<" * "<<tensor.typeAsStr()<<" = "<<result.typeAsStr();
        throw std::runtime_error(ss.str());
    }
    for (int i = 0; i < result.dim.first; i++) {
        for (int j = 0; j < result.dim.second; j++) {
            result.t[i][j] = 0.0;
            for (int k = 0; k < dim.second; k++) result.t[i][j] += t[i][k] * tensor.t[k][j];
        }
    }
}

Tensor& Tensor::operator=(const Tensor& tensor) {
    if (dim != tensor.dim) {
        dim = tensor.dim;
        t.resize(dim.first);
        for (int i = 0; i < dim.first; i++) t[i].resize(dim.second);
    }
    for (int i = 0; i < dim.first; i++) t[i] = tensor.t[i];
    return *this;
}

Tensor Tensor::operator+(const Tensor& tensor) const {
    Tensor result(Dim(dim.first, tensor.dim.second));
    add(tensor, result);
    return result;
}

Tensor Tensor::operator*(const Tensor& tensor) const {
    Tensor result(Dim(dim.first, tensor.dim.second));
    multiply(tensor, result);
    return result;
}

void Tensor::print(std::ostream& stream) const {
    stream<<"[ ";
    for (int i = 0; i < dim.first; i++) {
        if (i > 0) stream<<"  ";
        for (int j = 0; j < dim.second; j++) stream<<t[i][j]<<" ";
        if (i < dim.first - 1) stream<<std::endl;
    }
    stream<<"]"<<std::endl;
}

std::string Tensor::typeAsStr() const {
    std::stringstream ss;
    ss<<"("<<dim.first<<", "<<dim.second<<")";
    return ss.str();
}

}