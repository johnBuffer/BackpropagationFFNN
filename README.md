# BackpropagationFFNN

Simple backpropagation algorithm wrote from scratch in order to understand it.

## Usage

Declare network architecture: 1 input, 3 units in hidden layer, 1 output
```c++
std::vector<uint32_t> architecture{ 1, 3, 1 };
```

Create the **Network** and the **Optimizer** (with a training rate of 0.01)
```c++
ffnn::FFNeuralNetwork network(architecture);
ffnn::Optimizer optimizer(0.01f);
```
A simple struct to handle training examples
```c++
struct Example
{
	ffnn::Vectorf input;
  // The expected output
	ffnn::Vectorf output;
};
```
Generate training set, here we just want the output to be equal to the input
```c++
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
```
Train the network
```c++
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
```
View results
```c++
for (uint32_t i(10); i--;) {
  const float a = gen.getUnder(1.0f);
  const ffnn::Vectorf r = network.execute({ a });
  const float error = 0.5f * std::pow(a - r[0], 2);
  std::cout << "Result for " << a << ": " << r[0]  << " error: " << error << std::endl;
}
```

## Limitations
 - Currently just **sigmoid** activation is supported
 - Perf are probably very bad because of naive matrix operations
