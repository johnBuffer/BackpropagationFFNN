#include <iostream>
#include "number_generator.hpp"
#include "ffnn.hpp"


int main()
{
	std::vector<uint32_t> architecture{ 1, 3, 1 };

	ffnn::FFNeuralNetwork network(architecture);
	ffnn::Optimizer optimizer(0.01f);

	NumberGenerator gen;

	std::vector<ffnn::Vectorf> input_data;
	std::vector<ffnn::Vectorf> expected_data;

	uint64_t examples_count = 10000;
	for (uint64_t i(examples_count); i--;) {
		const float a = gen.getUnder(1.0f);
		input_data.push_back({ a });
		expected_data.push_back({ a });
	}

	const float threshold = 0.6f;
	float error = threshold + 1.0f;
	while (error > threshold) {
		error = 0.0f;
		for (uint64_t i(examples_count); i--;) {
			const float a = gen.getUnder(1.0f);
			std::vector<float> input{ a };
			std::vector<float> output{ a };
			optimizer.train(network, input, output);
			error += optimizer.error;
		}
		std::cout << error << std::endl;
	}
	

	for (uint32_t i(10); i--;) {
		const float a = gen.getUnder(1.0f);
		const ffnn::Vectorf r = network.execute({ a });
		const float error = 0.5f * std::pow(a - r[0], 2);
		std::cout << "Result for " << a << " " << r[0]  << " error " << error << std::endl;
	}

	return 0;
}