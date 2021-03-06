#include <iostream>
#include "number_generator.hpp"
#include "ffnn.hpp"

struct Example
{
	ffnn::Vectorf input;
	ffnn::Vectorf output;
};

int main()
{
	// The network architecture: 1 input, 3 units in hidden layer, 1 output
	std::vector<uint32_t> architecture{ 1, 3, 1 };
	// Create the network
	ffnn::FFNeuralNetwork network(architecture);
	// Create an optimizer
	ffnn::Optimizer optimizer(0.01f);

	// Create training set
	NumberGenerator gen;
	std::vector<Example> training_set;
	uint64_t examples_count = 10000;
	for (uint64_t i(examples_count); i--;) {
		const float a = gen.getUnder(1.0f);
		training_set.push_back({
			{a},
			{a}
		});
	}

	// Train the network
	const float threshold = 1.0f;
	float error = threshold + 1.0f;
	while (error > threshold) {
		error = 0.0f;
		for (const Example& example : training_set) {
			optimizer.train(network, example.input, example.output);
			error += optimizer.error;
		}
		std::cout << error << std::endl;
	}
	
	// View results
	for (uint32_t i(10); i--;) {
		const float a = gen.getUnder(1.0f);
		const ffnn::Vectorf r = network.execute({ a });
		const float error = 0.5f * std::pow(a - r[0], 2);
		std::cout << "Result for " << a << ": " << r[0]  << " error: " << error << std::endl;
	}

	return 0;
}