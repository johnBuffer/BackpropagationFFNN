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
	};

	struct Optimizer
	{
		float learning_rate;
		float error;

		Optimizer(float eta)
			: learning_rate(eta)
			, error(1.0f)
		{}

		void train(FFNeuralNetwork& network, Vector const& input, Vector const& expected_output)
		{
			const uint64_t depth = network.getDepth();
			const std::vector<Layer>& layers = network.layers;
			// Delta Weights -> the correction to apply to weights after this pass
			std::vector<Matrix> dw(depth-1);
			// Delta Bias -> the correction to apply to Bias after this pass
			std::vector<Matrix> db(depth-1);
			// Forward pass
            Vector const& output = network.execute(input);
			// Compute error for the last layer
			Vector const dedo = output - expected_output;
			error = 0.5f * dedo.dot(dedo);
			Vector const delta = dedo * getActivationDerivative(output, sigm_derivative);
			dw.back() = learning_rate * delta * transpose(crget(layers, 1).values);
			rget(db) = learning_rate * delta;
			// Propagate error in previous layers
			for (uint64_t i(1); i < depth - 1; ++i) {
				delta = crget(layers, i - 1).weights.transpose() * delta * getActivationDerivative(crget(layers, i).values);
				rget(dw, i) = learning_rate * delta * transpose(crget(layers, i + 1).values);
				rget(db, i) = learning_rate * delta;
			}
			// Update weights and bias
			updateNetwork(network, db, dw);
		}

		// For sigmoid
		static Eigen::VectorXf getActivationDerivative(Eigen::VectorXf const& v, ActivationFunction function)
		{
			return v.unaryExpr(function);
		}

        void correctLayer(uint32_t i, Vector const& last)
        {

        }

		void updateNetwork(ffnn::FFNeuralNetwork& network, const std::vector<Eigen::VectorXf>& bias_update, const std::vector<Matrix>& weights_update)
		{
			const uint64_t depth = network.getDepth();
			for (uint64_t i(1); i < depth; ++i) {
				Layer& layer = network.layers[i];
				// Update weights
				Matrix& w = layer.weights;
				w = w - weights_update[i-1];
				// Update bias
				Vector& b = layer.biases;
				b = b - bias_update[i-1];
			}
		}
	};
}
