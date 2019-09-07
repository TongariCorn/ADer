#include <iostream>
#include <cmath>

#include "../ader/variable.hpp"

using namespace std;

int main() {
    ader::Tensor xt(ader::Dim(1, 1));
    xt(0,0) = 3;
    ader::Variable x(xt);
    ader::Tensor yt(ader::Dim(1, 1));
    yt(0,0) = 2;
    ader::Variable y(yt);
    auto sinx3 = sin(x * x * y);
    sinx3.backprop(ader::Dim(0, 0));
    cout<<"(sin(x^2y))'|x=3,y=2 = "<<x.getGradient();
    cout<<"2xycos(x^2y)|x=3,y=2 = "<<(2 * xt * yt * (xt * xt * yt).cos())<<endl;


    // 行列演算
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
    cout<<"v1 = "<<endl<<v1;
    cout<<"v2 = "<<endl<<v2;
    cout<<"v3 = v1 * v2 = "<<endl<<v3<<endl;

    v3.backprop(ader::Dim(0, 0));
    cout<<"d(v1 * v2)_00/v2 = "<<endl;;
    v2.getGradient().print(cout);

    return 0;
}