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
    Tensor gradient;
    std::vector<std::shared_ptr<Node>> nodes;

    virtual void calcGradient() = 0;

    void addNodesToQ(std::queue<std::shared_ptr<Node>>& q);

public:
    Node(Dim dim = Dim(0,0)) : value(dim), gradient(dim) {}
    
    const Tensor& getValue() { return value; }
    const Tensor& getGradient() { return gradient; }

    void clearGradient();
    void addGradient(const Tensor& tensor) { gradient = gradient + tensor; }
    void backprop();

    void print(std::ostream& stream) const {
        value.print(stream);
    }

    virtual ~Node() {}
};

class ConstNode : public Node {
    void calcGradient() {}

public:
    ConstNode(Dim dim = Dim(0,0)) : Node(dim) {}
    ConstNode(const Tensor& tensor);

    ~ConstNode() {}
};

class AdderNode : public Node {
    void calcGradient();

public:
    AdderNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2);

    ~AdderNode() {}
};

class MultiplierNode : public Node {
    void calcGradient();

public:
    MultiplierNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2);

    ~MultiplierNode() {}
};

}

#endif