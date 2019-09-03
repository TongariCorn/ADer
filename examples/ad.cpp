#include <iostream>

#include "../ader/variable.hpp"

using namespace std;

int main() {
    ader::Variable x;
    x + x;
    ader::Tensor t(ader::Dim(2, 2));
    t[ader::Dim(0, 0)] = 1;
    t[ader::Dim(0, 1)] = 3;
    t[ader::Dim(1, 0)] = 0;
    t[ader::Dim(1, 1)] = 1;

    ader::Tensor t2(ader::Dim(2, 2));
    t2[ader::Dim(0, 0)] = 2;
    t2[ader::Dim(0, 1)] = 0;
    t2[ader::Dim(1, 0)] = 1;
    t2[ader::Dim(1, 1)] = 1;

    auto v1 = ader::Variable(t);
    auto v2 = ader::Variable(t2);
    auto t3 = v1 + v2;
    v1.getValue().print(cout);
    v2.print(cout);
    t3.print(cout);

    return 0;
}