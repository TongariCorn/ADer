#include "node.hpp"

namespace ader {

void Node::clearGradient() {
    gradient.clear();
    for (auto& node : nodes) node->clearGradient();
}

void Node::addNodesToQ(std::queue<std::shared_ptr<Node>>& q) {
    for (auto& node : nodes) q.push(node);
}

void Node::backprop() {
    // Breadth First
    std::queue<std::shared_ptr<Node>> q;
    calcGradient();
    addNodesToQ(q);
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        node->calcGradient();
        node->addNodesToQ(q);
    }
}

ConstNode::ConstNode(const Tensor& tensor) : Node(tensor.getDim()) {
    value = tensor;
    gradient = Tensor(value.getDim());
}

AdderNode::AdderNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2) {
    if (n1 == nullptr || n2 == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(2);
    nodes[0] = n1;
    nodes[1] = n2;
    value = nodes[0]->getValue() + nodes[1]->getValue();
    gradient = Tensor(value.getDim());
}

void AdderNode::calcGradient() {
    nodes[0]->addGradient(gradient);
    nodes[1]->addGradient(gradient);
}

MultiplierNode::MultiplierNode(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2) {
    if (n1 == nullptr || n2 == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(2);
    nodes[0] = n1;
    nodes[1] = n2;
    value = nodes[0]->getValue() * nodes[1]->getValue();
    gradient = Tensor(value.getDim());
}

void MultiplierNode::calcGradient() {
    nodes[0]->addGradient(gradient * nodes[1]->getValue().transpose());
    nodes[1]->addGradient(nodes[0]->getValue().transpose() * gradient);
}

}