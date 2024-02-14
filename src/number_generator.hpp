#pragma once

#include <random>
#include <limits>
#include <memory>


struct NumberGenerator
{
	NumberGenerator(bool random_seed = true)
		: distribution(-1.0f, 1.0f)
	{
		if (random_seed) {
			gen = std::mt19937(rd());
		}
		else {
			gen = std::mt19937(0);
		}
	}

	float get(float range = 1.0f)
	{
		return range * distribution(gen);
	}

	float getUnder(float max_value)
	{
		return (distribution(gen) + 1.0f) * 0.5f * max_value;
	}

	float getMaxRange()
	{
		return std::numeric_limits<float>::max() * distribution(gen);
	}

	void reset(bool use_seed)
	{
		// TODO add random seed case
		gen = std::mt19937(use_seed ? rd() : 0);
		distribution.reset();
	}

	std::uniform_real_distribution<float> distribution;
	std::random_device rd;
	std::mt19937 gen;

	static std::unique_ptr<NumberGenerator> s_instance;
};

