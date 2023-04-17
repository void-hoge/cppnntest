#pragma once

#include "activation.hpp"
#include "layer.hpp"

class Model {
private:
	std::size_t inputsize;
	std::size_t outputsize;
	std::vector<Layer::Dense> model;
	std::string m_loss;
	std::vector<std::vector<double>> diff_error;

	double mean_squared_error(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& t);

	std::vector<std::vector<double>> mean_squared_error_back(std::vector<std::vector<double>>& absolute_error);
	
	double mean_cross_entropy_error(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& t);

	std::vector<std::vector<double>> mean_cross_entropy_error_back(std::vector<std::vector<double>>& error);

	double caluculate_loss(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& t);

	std::vector<std::vector<double>> numerical_gradient_weight(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, std::size_t idx);
	
	std::vector<double> numerical_gradient_bias(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, std::size_t idx);

	void backward();
public:
	Model(std::size_t inputsize);
	void add_dense_layer(std::size_t unit, ActivationType act);
	std::vector<std::vector<double>> predict(std::vector<std::vector<double>>& data);
	std::vector<double> fit(
		std::size_t step, double learningrate,
		std::vector<std::vector<double>>& x,
		std::vector<std::vector<double>>& y,
		std::size_t batchsize, std::string loss);
	std::vector<double> fitbp(
		std::size_t step, double learningrate,
		std::vector<std::vector<double>>& x,
		std::vector<std::vector<double>>& y,
		std::size_t batchsize, std::string loss);
};
