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
    Variable(const Tensor& tensor) : node(std::make_shared<ConstNode>(tensor)) {}
    Variable(std::shared_ptr<Node> n) : node(n) {}

    friend Variable operator+(const Variable&, const Variable&);

    const Tensor& getValue() {
        return node->getValue();
    }

    void print(std::ostream& stream) {
        node->print(stream);
    }
};

Variable operator+(const Variable& v1, const Variable& v2) {
    std::shared_ptr<Node> n = std::make_shared<AdderNode>(v1.node, v2.node);
    return Variable(n);
}

}

#endif