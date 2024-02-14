 #pragma once

#include "mat.hpp"
#include <cmath>
#include <iostream>
#include "number_generator.hpp"


namespace ffnn
{
	float sigm(float x)
	{
		return 1.0f / (1.0f + exp(-x));
	}

	struct Layer
	{
		Matrixf weights;
		Vectorf values;
		Vectorf bias;

		Layer(uint64_t size, uint64_t prev_size)
			: weights(prev_size, size)
			, values(size)
			, bias(bool(prev_size) * size)
		{
			NumberGenerator gen;
			for (float& f : weights.values) {
				f = gen.get();
			}

			for (float& f : bias) {
				f = gen.get();
			}
		}

		uint64_t getSize() const
		{
			return values.size();
		}

		void computeValues(const Vectorf& prev_values)
		{
			values = map(sigm, weights * prev_values + bias);
		}
	};

	struct FFNeuralNetwork
	{
		std::vector<Layer> layers;

		FFNeuralNetwork(const std::vector<uint32_t>& architecture)
		{
			if (!architecture.empty()) {
				layers.emplace_back(architecture.front(), 0);
				for (uint64_t i(1); i < architecture.size(); ++i) {
					addLayer(architecture[i]);
				}
			}
		}

		void addLayer(uint64_t size)
		{
			layers.emplace_back(size, layers.back().getSize());
		}

		const Vectorf& execute(const Vectorf& input)
		{
			const uint64_t depth = getDepth();
			layers[0].values = input;
			for (uint64_t i(1); i < depth; ++i) {
				layers[i].computeValues(layers[i - 1].values);
			}
			return rget(layers).values;
		}

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
		{
		}

		void train(FFNeuralNetwork& network, const Vectorf& input, const Vectorf& expected_output)
		{
			const uint64_t depth = network.getDepth();
			const std::vector<Layer>& layers = network.layers;
			// Delta Weights -> the correction to apply to weights after this pass
			std::vector<Matrixf> dw(depth-1);
			// Delta Bias -> the correction to apply to Bias after this pass
			std::vector<Vectorf> db(depth-1);
			// Forward pass
			network.execute(input);
			// Compute error for the last layer
			const Vectorf& output = crget(layers).values;
			const Vectorf dedo = output - expected_output;
			error = 0.5f * dot(dedo, dedo);
			Vectorf delta = dedo * getActivationDerivative(output);
			rget(dw) = learning_rate * delta * transpose(crget(layers, 1).values);
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
		static Vectorf getActivationDerivative(const Vectorf& v)
		{
			return v * (1.0f - v);
		}

		void updateNetwork(ffnn::FFNeuralNetwork& network, const std::vector<Vectorf>& bias_update, const std::vector<Matrixf>& weights_update)
		{
			const uint64_t depth = network.getDepth();
			for (uint64_t i(1); i < depth; ++i) {
				Layer& layer = network.layers[i];
				// Update weights
				Matrixf& w = layer.weights;
				w = w - weights_update[i-1];
				// Update bias
				Vectorf& b = layer.bias;
				b = b - bias_update[i-1];
			}
		}
	};
}
