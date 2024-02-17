#include <iostream>
#include "ffnn.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
    ffnn::FFNeuralNetwork network({2, 2, 2});
    network.setWeights(1, 0, {0.3f, -0.4f});
    network.setWeights(1, 1, {0.2f, 0.6f});

    network.setWeights(2, 0, {0.7f, 0.5f});
    network.setWeights(2, 1, {-0.3f, -0.1f});

    network.setBias(0, 0, 0.0f);
    network.setBias(0, 1, 0.0f);

    network.setBias(1, 0, 0.25f);
    network.setBias(1, 1, 0.45f);

    network.setBias(2, 0, 0.15f);
    network.setBias(2, 1, 0.35f);

    //network.print();

    ffnn::Optimizer optimizer(0.5f);

    for (uint32_t i{0}; i < 1000; ++i) {
        optimizer.train(network, Eigen::VectorXf{{2.0f, 3.0f}}, Eigen::VectorXf{{1.0f, 0.2f}});
    }

    return 0;
}
