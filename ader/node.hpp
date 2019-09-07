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
    friend std::ostream& operator<<(std::ostream&, const std::shared_ptr<Node>);

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

class ConstMultiplierNode : public Node {
    double c;

    void calcGradient();

public:
    ConstMultiplierNode(double d, std::shared_ptr<Node> n);

    ~ConstMultiplierNode() {}
};

class SinNode : public Node {
    void calcGradient();

public:
    SinNode(std::shared_ptr<Node> n);

    ~SinNode() {}
};

class CosNode : public Node {
    void calcGradient();

public:
    CosNode(std::shared_ptr<Node> n);

    ~CosNode() {}
};

class SquareNode : public Node {
    void calcGradient();

public:
    SquareNode(std::shared_ptr<Node> n);

    ~SquareNode() {}
};

class SumNode : public Node {
    void calcGradient();

public:
    SumNode(std::shared_ptr<Node> n);

    ~SumNode() {}
};

}

#endif