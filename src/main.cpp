#include <iostream>
#include "ffnn.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
    ffnn::FFNeuralNetwork network({2, 2, 2});
    network.setWeights(1, 0, {0.15f, 0.2f});
    network.setWeights(1, 1, {0.25f, 0.3f});

    network.setWeights(2, 0, {0.4f, 0.45f});
    network.setWeights(2, 1, {0.5f, 0.55f});

    network.setBias(0, 0, 0.0f);
    network.setBias(0, 1, 0.0f);

    network.setBias(1, 0, 0.35f);
    network.setBias(1, 1, 0.35f);

    network.setBias(2, 0, 0.6f);
    network.setBias(2, 1, 0.6f);

    //network.print();

    ffnn::Optimizer optimizer(0.5f);
    optimizer.train(network, Eigen::VectorXf{{0.05f, 0.1f}}, Eigen::VectorXf{{0.01f, 0.99f}});

    Eigen::VectorXf v1{{1.0f, 2.0f}};
    Eigen::VectorXf v2{{3.0f, 4.0f}};
    std::cout << v1 * v2.transpose() << std::endl;
    std::cout << v1.transpose() * v2 << std::endl;

    return 0;
}
