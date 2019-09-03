#ifndef ADER_VARIABLE_HPP
#define ADER_VARIABLE_HPP

#include <iostream>
#include <memory>

#include "node.hpp"

namespace ader {

class Variable {
    std::shared_ptr<Node> node;
public:
    Variable() : node(std::make_shared<VariableNode>()) {}
    Variable(const Tensor& tensor) : node(std::make_shared<VariableNode>(tensor)) {}
    Variable(std::shared_ptr<Node> n) : node(n) {}

    void assign(const Tensor& tensor) {
        std::shared_ptr<Node> v = std::make_shared<VariableNode>();
        node.swap(v);
        node->swapNextNodes(v);
        std::dynamic_pointer_cast<VariableNode>(v)->assignValue(tensor);
    }

    Variable& operator=(const Tensor& tensor) {
        assign(tensor);
        return *this;
    }

    Variable operator+(Variable& variable) {
        std::shared_ptr<Node> n = std::make_shared<AdderNode>(node, variable.node);
        node->addNextNode(n);
        if (variable.node != node) variable.node->addNextNode(n);
        return Variable(n);
    }

    const Tensor& getValue() {
        return node->getValue();
    }

    void print(std::ostream& stream) {
        node->print(stream);
    }
};

}

#endif