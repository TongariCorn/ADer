#include <iostream>

#include "../ader/variable.hpp"

using namespace std;

int main() {
    ader::Variable x;
    x + x;
    ader::Tensor t1(ader::Dim(2, 2));
    t1(0,0) = 1;
    t1(0,1) = 3;
    t1(1,0) = 0;
    t1(1,1) = 1;

    ader::Tensor t2(ader::Dim(2, 2));
    t2(0,0) = 2;
    t2(0,1) = 0;
    t2(1,0) = 1;
    t2(1,1) = 1;

    auto v1 = ader::Variable(t1);
    auto v2 = ader::Variable(t2);
    auto v3 = v1 + v2;
    v1.getValue().print(cout);
    v2.print(cout);
    v3.print(cout);

    v1 = v2.getValue();
    v1.getValue().print(cout);
    v3.getValue().print(cout);

    return 0;
}