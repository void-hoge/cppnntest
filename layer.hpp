#pragma once

#include "activation.hpp"

#include <vector>
#include <memory>

namespace Layer{

class Dense {
private:
	std::shared_ptr<Activation> activation;
	std::vector<std::vector<double>> lastdata;
public:
	std::vector<std::vector<double>> grad_weights;
	std::vector<double> grad_biases;
	std::vector<double> biases;
	std::vector<std::vector<double>> weights;
	Dense(
		std::size_t input,
		std::size_t output,
		ActivationType acttype);
	std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& data);
	std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& data);
};

} // namespace Layer
