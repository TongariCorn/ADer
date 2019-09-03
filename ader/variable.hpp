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

    Variable operator+(Variable& variable) {
        std::shared_ptr<Node> n = std::make_shared<AdderNode>(node, variable.node);
        variable.node->addNextNode(n);
        return Variable(n);
    }

    const Tensor& getValue() const {
        return node->getValue();
    }

    void print(std::ostream& stream) const {
        node->print(stream);
    }
};

}

#endif