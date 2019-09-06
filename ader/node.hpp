#ifndef ADER_NODE_HPP
#define ADER_NODE_HPP

#include <memory>
#include <list>
#include <queue>

#include "math.hpp"

namespace ader {

class Node {
protected:
    Tensor value;

public:
    Node(Dim dim = Dim(0,0)) : value(dim) {}
    
    const Tensor& getValue() { return value; }

    void print(std::ostream& stream) const {
        value.print(stream);
    }

    virtual ~Node() {}
};

class AdderNode : public Node {
    std::shared_ptr<Node> node1, node2;
public:
    AdderNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2) {
        if (n1 == nullptr || n2 == nullptr) throw std::runtime_error("operation on an uninitialized variable");
        node1 = n1;
        node2 = n2;
        value = node1->getValue() + node2->getValue();
    }

    ~AdderNode() {}
};

class ConstNode : public Node {
public:
    ConstNode(Dim dim = Dim(0,0)) : Node(dim) {}
    ConstNode(const Tensor& tensor) : Node(tensor.getDim()) {
        value = tensor;
    }

    ~ConstNode() {}
};

}

#endif