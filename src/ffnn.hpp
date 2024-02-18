 #pragma once

#include <Eigen/Dense>
#include <iostream>


namespace ffnn
{
    using ActivationFunction = float(*)(float);
    using Matrix = Eigen::MatrixXf;
    using Vector = Eigen::VectorXf;

	float sigm(float x)
	{
		return 1.0f / (1.0f + exp(-x));
	}

    float sigm_derivative(float x)
    {
        return x * (1.0f - x);
    }

	struct Layer
	{
		Matrix weights;
		Vector values;
        Vector biases;

		Layer(uint64_t size, uint64_t prev_size)
			: weights(size, prev_size)
			, values(size)
			, biases(size)
		{
			weights.setRandom();
            biases.setRandom();
		}

        [[nodiscard]]
		uint64_t getSize() const
		{
			return values.size();
		}

		void computeValues(Vector const& prev_values)
		{
            // Compute weights and bias contributions
            Vector const sums = weights * prev_values + biases;
			values = sums.unaryExpr(&sigm);
		}
	};

	struct FFNeuralNetwork
	{
        uint32_t           input_count = 0;
		std::vector<Layer> layers;

        explicit
		FFNeuralNetwork(const std::vector<uint32_t>& architecture)
		{
			for (auto const layer_size : architecture) {
                addLayer(layer_size);
            }
		}

        void setWeights(uint32_t layer_idx, uint32_t neuron_idx, std::vector<float> const& weights)
        {
            auto& l = layers[layer_idx];
            auto const size = weights.size();
            for (int64_t i{0}; i < size; ++i) {
                l.weights(neuron_idx, i) = weights[i];
            }
        }

        void setBias(uint32_t layer_idx, uint32_t neuron_idx, float bias)
        {
            layers[layer_idx].biases[neuron_idx] = bias;
        }

		void addLayer(uint64_t size)
		{
            if (layers.empty()) {
                input_count = size;
                layers.emplace_back(size, 0);
            } else {
                layers.emplace_back(size, layers.back().getSize());
            }
		}

		Vector const& execute(Vector const& input)
		{
            assert(!layers.empty());
            assert(input.size() == input_count);

			uint64_t const depth = getDepth();
            layers[0].values = input;
			for (uint64_t i(1); i < depth; ++i) {
				layers[i].computeValues(layers[i - 1].values);
			}
			return layers.back().values;
		}

        [[nodiscard]]
		uint64_t getDepth() const
		{
			return layers.size();
		}

        void print() const
        {
            uint32_t i{0};
            for (auto const& l : layers) {
                auto const curr_size = l.weights.rows();
                auto const prev_size = l.weights.cols();
                std::cout << "[ Layer " << i << " ]" << std::endl;
                for (uint64_t k{0}; k < curr_size; ++k) {
                    std::cout << "    Neuron [" << k << "]: ";
                    for (uint64_t n_idx{0}; n_idx < prev_size; ++n_idx) {
                        std::cout << l.weights(k, n_idx) << " ";
                    }
                    std::cout << " Bias: " << l.biases[k] << std::endl;
                }
                ++i;
            }
        }
	};

	struct Optimizer
	{
		float learning_rate;
		float error;

        explicit
		Optimizer(float eta)
			: learning_rate(eta)
			, error(1.0f)
		{}

		void train(FFNeuralNetwork& network, Vector const& input, Vector const& expected_output)
		{
			uint64_t const      depth  = network.getDepth();
			std::vector<Layer>& layers = network.layers;
			// Delta Weights -> the correction to apply to weights after this pass
			std::vector<Matrix> dw(depth);
			// Delta Bias -> the correction to apply to Bias after this pass
			std::vector<Vector> db(depth);
			// Forward pass
            Vector const& output = network.execute(input);
            Vector dCdO = 2.0f * (output - expected_output);
            for (uint64_t i{depth}; --i;) {
                Layer& current_layer = layers[i];
                Vector dCdI = dCdO.cwiseProduct(current_layer.values.unaryExpr(&sigm_derivative));
                Matrix dCdW = dCdI * layers[i-1].values.transpose();
                dw[i] = dCdW;
                db[i] = dCdI;
                dCdO = current_layer.weights * dCdI;
            }
            updateNetwork(network, dw, db);
		}

		void updateNetwork(ffnn::FFNeuralNetwork& network, std::vector<Matrix> const& weights_update, std::vector<Vector> const& bias_update)
		{
			const uint64_t depth = network.getDepth();
			for (uint64_t i(1); i < depth; ++i) {
				Layer& layer = network.layers[i];
				// Update weights
				Matrix& w = layer.weights;
				w = w - learning_rate * weights_update[i];
				// Update bias
				Vector& b = layer.biases;
				b = b - learning_rate * bias_update[i];
			}
		}
	};
}
