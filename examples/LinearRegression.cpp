#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include "../ader/variable.hpp"

using namespace std;

#define SAMPLE_NUM 30

int main() {
    random_device rnd;
    mt19937 engine(rnd());

    // z = 2x + 3y - 10にN(0, 5)に従うノイズを乗せたサンプルを用意
    uniform_real_distribution<> dist1(-10.0, 10.0);
    normal_distribution<>       dist2(0.0, 5.0);
    ader::Tensor xyt(ader::Dim(3, SAMPLE_NUM));
    ader::Tensor zt(ader::Dim(1, SAMPLE_NUM));
    for (int i = 0; i < SAMPLE_NUM; i++) {
        double x = dist1(engine);
        double y = dist1(engine);
        xyt(0, i) = x;
        xyt(1, i) = y;
        xyt(2, i) = 1.0;
        zt(0, i) = 2.0*x + 3.0*y - 10.0 + dist2(engine);
        cout<<"sample["<<i<<"] : "<<zt(0, i)<<", ("<<x<<", "<<y<<")"<<endl;
    }

    // 自動微分によって残差平方和を最小にするように最降降下法を行う
    ader::Variable W(ader::Dim(1, 3));
    ader::Variable XY(xyt);
    ader::Variable Z(zt);
    for (int i = 0; i < 100; i++) {
        auto loss = ader::sum(ader::square(W * XY + (-1.0 * Z)));

        loss.backprop();
        W = W + (-0.0005 * W.getGradient());
        if (i % 10 == 0) {
            cout<<"i = "<<i<<": loss = "<<loss;
            cout<<"        W = "<<W<<endl;
        }
    }

    return 0;
}