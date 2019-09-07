#include <iostream>

#include "../ader/variable.hpp"

using namespace std;

int main() {
    ader::Tensor t(ader::Dim(1, 1));
    t(0,0) = 3;
    ader::Variable x(t);
    auto x2 = x * x;
    x2.backprop(ader::Dim(0, 0));
    x.getGradient().print(std::cout);


    ader::Tensor t1(ader::Dim(2, 2));
    t1(0,0) = 1;
    t1(0,1) = 3;
    t1(1,0) = 0;
    t1(1,1) = 1;

    ader::Tensor t2(ader::Dim(2, 1));
    t2(0,0) = 2;
    t2(1,0) = 1;

    auto v1 = ader::Variable(t1);
    auto v2 = ader::Variable(t2);
    auto v3 = v1 * v2;
    v1.getValue().print(cout);
    v2.print(cout);
    v3.print(cout);

    v3.backprop(ader::Dim(0, 0));
    v2.getGradient().print(std::cout);

    return 0;
}