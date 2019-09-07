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

std::ostream& operator<<(std::ostream& stream, const std::shared_ptr<Node> node) {
    node->print(stream);
    return stream;
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

ConstMultiplierNode::ConstMultiplierNode(double d, std::shared_ptr<Node> n) {
    if (n == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(1);
    nodes[0] = n;
    c = d;
    value = c * nodes[0]->getValue();
    gradient = Tensor(value.getDim());
}

void ConstMultiplierNode::calcGradient() {
    nodes[0]->addGradient(c * gradient);
}

SinNode::SinNode(std::shared_ptr<Node> n) {
    if (n == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(1);
    nodes[0] = n;
    value = nodes[0]->getValue().sin();
    gradient = Tensor(value.getDim());
}

void SinNode::calcGradient() {
    nodes[0]->addGradient(gradient.hadamard(nodes[0]->getValue().cos()));
}

CosNode::CosNode(std::shared_ptr<Node> n) {
    if (n == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(1);
    nodes[0] = n;
    value = nodes[0]->getValue().cos();
    gradient = Tensor(value.getDim());
}

void CosNode::calcGradient() {
    nodes[0]->addGradient(gradient.hadamard(-nodes[0]->getValue().sin()));
}

SquareNode::SquareNode(std::shared_ptr<Node> n) {
    if (n == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(1);
    nodes[0] = n;
    value = nodes[0]->getValue().hadamard(nodes[0]->getValue());
    gradient = Tensor(value.getDim());
}

void SquareNode::calcGradient() {
    nodes[0]->addGradient(2 * gradient.hadamard(nodes[0]->getValue()));
}

SumNode::SumNode(std::shared_ptr<Node> n) {
    if (n == nullptr) throw std::runtime_error("operation on an uninitialized variable");
    nodes.resize(1);
    nodes[0] = n;
    value = Tensor(Dim(1, 1));
    value(0, 0) = nodes[0]->getValue().sum();
    gradient = Tensor(value.getDim());
}

void SumNode::calcGradient() {
    nodes[0]->addGradient(Tensor(nodes[0]->getValue().getDim(), gradient(0, 0)));
}

}