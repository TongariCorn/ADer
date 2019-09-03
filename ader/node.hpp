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
    std::list<std::weak_ptr<Node>> nextNodes;
    bool dirtyBit = false;

    // valueの値が最新ではない事を示す
    void markDirtyBit() {
        dirtyBit = true;
        for (auto it = begin(nextNodes); it != end(nextNodes); ) {
            if (it->expired()) it = nextNodes.erase(it);
            else {
                it->lock()->markDirtyBit();
                it++;
            }
        }
    }

public:
    Node(Dim dim = Dim(0,0)) : value(dim) {}
    
    const Tensor& getValue() {
        value.print(std::cout);
        std::cout<<nextNodes.size()<<std::endl;
        if (dirtyBit) calcValue();
        return value;
    }
    virtual void calcValue() = 0;

    void addNextNode(std::shared_ptr<Node> node) {
        nextNodes.emplace_front(node);
    }

    void swapNextNodes(std::shared_ptr<Node> node) {
        nextNodes.swap(node->nextNodes);
    }

    void print(std::ostream& stream) const {
        value.print(stream);
    }

    virtual ~Node() {}
};

class FunctionNode : public Node {
public:
    virtual ~FunctionNode() {}
};

class AdderNode : public FunctionNode {
    std::shared_ptr<Node> node1, node2;
public:
    AdderNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2) {
        node1 = n1;
        node2 = n2;
        calcValue();
    }

    void calcValue() {
        if (node1 == nullptr || node2 == nullptr) throw std::runtime_error("operation on an uninitialized variable");
        value = node1->getValue() + node2->getValue();
        dirtyBit = false;
    }

    ~AdderNode() {}
};

class VariableNode : public Node {
public:
    VariableNode(Dim dim = Dim(0,0)) : Node(dim) {}
    VariableNode(const Tensor& tensor) : Node(tensor.getDim()) {
        assignValue(tensor);
    }

    void calcValue() { dirtyBit = false; }
    void assignValue(const Tensor& tensor) { 
        value = tensor;
        markDirtyBit();
    }

    ~VariableNode() {}
};

}

#endif