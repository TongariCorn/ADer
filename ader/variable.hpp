#ifndef ADER_VARIABLE_HPP
#define ADER_VARIABLE_HPP

#include <iostream>
#include <memory>

#include "node.hpp"

namespace ader {

class Variable {
    std::shared_ptr<Node> node;
public:
    Variable() : node(std::make_shared<ConstNode>()) {}
    Variable(Dim dim) : node(std::make_shared<ConstNode>(dim)) {}
    Variable(const Tensor& tensor) : node(std::make_shared<ConstNode>(tensor)) {}
    Variable(std::shared_ptr<Node> n) : node(n) {}

    friend Variable operator+(const Variable&, const Variable&);
    friend Variable operator-(const Variable&, const Variable&);
    friend Variable operator*(const Variable&, const Variable&);
    friend Variable operator*(double, const Variable&);
    friend Variable sin(const Variable&);
    friend Variable cos(const Variable&);
    friend Variable square(const Variable&);
    friend Variable sum(const Variable&);

    const Tensor& getValue() {
        return node->getValue();
    }

    const Tensor& getGradient() {
        return node->getGradient();
    }

    void backprop(Dim dim = Dim(0,0)) {
        node->clearGradient();

        auto grad = Tensor(node->getGradient().getDim());
        grad(dim.first, dim.second) = 1.0;
        node->addGradient(grad);
        node->backprop();
    }

    friend std::ostream& operator<<(std::ostream&, const Variable&);
    void print(std::ostream& stream) const {
        node->print(stream);
    }
};

Variable operator+(const Variable& v1, const Variable& v2) {
    std::shared_ptr<Node> n = std::make_shared<AddNode>(v1.node, v2.node);
    return Variable(n);
}

Variable operator-(const Variable& v1, const Variable& v2) {
    std::shared_ptr<Node> n = std::make_shared<SubNode>(v1.node, v2.node);
    return Variable(n);
}

Variable operator*(const Variable& v1, const Variable& v2) {
    std::shared_ptr<Node> n = std::make_shared<MulNode>(v1.node, v2.node);
    return Variable(n);
}

Variable operator*(double d, const Variable& v) {
    std::shared_ptr<Node> n = std::make_shared<ConstMulNode>(d, v.node);
    return Variable(n);
}

Variable sin(const Variable& v) {
    std::shared_ptr<Node> n = std::make_shared<SinNode>(v.node);
    return Variable(n);
}

Variable cos(const Variable& v) {
    std::shared_ptr<Node> n = std::make_shared<CosNode>(v.node);
    return Variable(n);
}

Variable square(const Variable& v) {
    std::shared_ptr<Node> n = std::make_shared<SquareNode>(v.node);
    return Variable(n);
}

Variable sum(const Variable& v) {
    std::shared_ptr<Node> n = std::make_shared<SumNode>(v.node);
    return Variable(n);
}

std::ostream& operator<<(std::ostream& stream, const Variable& v) {
    v.print(stream);
    return stream;
}

}

#endif