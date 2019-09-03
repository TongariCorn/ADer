#ifndef ADER_NODE_HPP
#define ADER_NODE_HPP

#include <memory>
#include <list>

#include "math.hpp"

namespace ader {

class Node {
protected:
    Tensor value;
    std::list<std::weak_ptr<Node>> nextNodes;

public:
    Node(Dim dim = Dim(0,0)) : value(dim) {}
    
    const Tensor& getValue() const { return value; }
    void addNextNode(std::shared_ptr<Node> node) {
        nextNodes.emplace_front(node);
    }

    void print(std::ostream& stream) const {
        value.print(stream);
    }
};

class VariableNode : public Node {
public:
    VariableNode(Dim dim = Dim(0,0)) : Node(dim) {}
    VariableNode(const Tensor& tensor) : Node(tensor.getDim()) {
        value = tensor;
    }

    void assignValue(const Tensor& tensor) { value = tensor; }
};

class FunctionNode : public Node {
public:
};

class AdderNode : public FunctionNode {
    std::shared_ptr<Node> node1, node2;
public:
    AdderNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2) {
        if (n1 == nullptr || n2 == nullptr) throw std::runtime_error("operation on an uninitialized variable");
        node1 = n1;
        node2 = n2;
        value = node1->getValue() + node2->getValue();
    }
};

}

#endif