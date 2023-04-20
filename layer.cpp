#include "layer.hpp"

#include <vector>
#include <cmath>
#include <random>
#include <sstream>

Layer::Dense::Dense(
	std::size_t input,
	std::size_t output,
	ActivationType act):
	biases(output, 0),
	weights(output, std::vector<double>(input, 0)) {
	double sigma;
	switch (act) {
	case ActivationType::relu:
		sigma = std::sqrt((double)2.0/input);
		break;
	case ActivationType::sigmoid:
	case ActivationType::softmax:
		sigma = std::sqrt((double)1.0/input);
		break;
	default:
		sigma = 0.05;
		break;
	}
	std::random_device seed;
	std::mt19937 rng(seed());
	std::normal_distribution<> normaldist(0.0, sigma);
	for (std::size_t i = 0; i < output; i++) {
		this->biases[i] = normaldist(rng);
		for (std::size_t j = 0; j < input; j++) {
			this->weights[i][j] = normaldist(rng);
		}
	}
	switch(act) {
	case ActivationType::relu:
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<ReLU>(new ReLU()));
		break;
	case ActivationType::sigmoid:
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<Sigmoid>(new Sigmoid()));
		break;
	case ActivationType::softmax:
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<SoftMax>(new SoftMax()));
		break;
	default:
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<Linear>(new Linear()));
		break;
	}
}


std::vector<std::vector<double>> Layer::Dense::forward(std::vector<std::vector<double>>& data) {
	this->lastdata = data;
	std::vector<std::vector<double>> mat;
	for (std::size_t idx = 0; auto& batch: data) {
		if (batch.size() != this->weights[0].size()) {
			std::stringstream ss;
			ss << "Unexpected data was taken." << std::endl
			   << "Expected data's size was " << this->weights[0].size() << ". "
			   << "But the data's size was " << batch.size() << "."
			   << std::endl;
			throw std::logic_error(ss.str());
		}
		std::vector<double> dst;
		for (std::size_t i = 0; i < this->weights.size(); i++) {
			double sum = 0;
			for (std::size_t j = 0; j < this->weights[i].size(); j++) {
				sum += this->weights[i][j]*batch[j];
			}
			sum -= this->biases[i];
			dst.push_back(sum);
		}
		mat.push_back(dst);
		idx++;
	}
	mat = this->activation->forward(mat);
	return mat;
}

std::vector<std::vector<double>> Layer::Dense::backward(std::vector<std::vector<double>>& data) {
	data = this->activation->backward(data);
	std::vector<std::vector<double>> mat;
	for (std::size_t i = 0; i < data.size(); i++) {
		std::vector<double> row;
		for (std::size_t j = 0; j < this->weights[0].size(); j++) {
			double t = 0;
			for (std::size_t k = 0; k < this->weights.size(); k++) {
				t += this->weights[k][j] * data[i][k];
			}
			row.push_back(t);
		}
		mat.push_back(row);
	}

	this->grad_weights.clear();
	for (std::size_t i = 0; i < this->weights.size(); i++) {
		std::vector<double> row;
		for (std::size_t j = 0; j < this->weights[i].size(); j++) {
			double t = 0;
			for (std::size_t k = 0; k < data.size(); k++) {
				t += data[k][i] * this->lastdata[k][j];
			}
			row.push_back(t);
		}
		this->grad_weights.push_back(row);
	}

	this->grad_biases.clear();
	for (std::size_t i = 0; i < this->biases.size(); i++) {
		double t = 0;
		for (std::size_t j = 0; j < data.size(); j++) {
			t += data[j][i];
		}
		this->grad_biases.push_back(t);
	}
	return mat;
}
