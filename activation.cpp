#include "activation.hpp"

Sigmoid::Sigmoid() {}

std::vector<std::vector<double>> Sigmoid::forward(std::vector<std::vector<double>>& x) {
	std::vector<std::vector<double>> mat;
	for (auto batch: x) {
		std::vector<double> row;
		for (auto num: batch) {
			row.push_back((double)1.0/(std::exp(-num)+1.0));
		}
		mat.push_back(row);
	}
	this->lastdata = mat;
	return mat;
}

std::vector<std::vector<double>> Sigmoid::backward(std::vector<std::vector<double>>& x) {
	std::vector<std::vector<double>> mat;
	for (std::size_t i = 0; i < x.size(); i++) {
		std::vector<double> row;
		for (std::size_t j = 0; j < x[i].size(); j++) {
			row.push_back(x[i][j] * (1.0-this->lastdata[i][j]) * this->lastdata[i][j]);
		}
		mat.push_back(row);
	}
	return mat;
}

Linear::Linear() {}

std::vector<std::vector<double>> Linear::forward(std::vector<std::vector<double>>& x) {
	this->lastdata = x;
	std::vector<std::vector<double>> mat;
	for (auto batch: x) {
		std::vector<double> row;
		for (auto num: batch) {
			row.push_back(num);
		}
		mat.push_back(row);
	}
	return mat;
}

std::vector<std::vector<double>> Linear::backward(std::vector<std::vector<double>>& x) {
	return x;
}

SoftMax::SoftMax() {}

std::vector<std::vector<double>> SoftMax::forward(std::vector<std::vector<double>>& x) {
	this->lastdata = x;
	std::vector<std::vector<double>> mat;
	for (auto batch: x) {
		std::vector<double> row;
		double maxelm = *max_element(batch.begin(), batch.end());
		double sum = 0;
		for (auto num: batch) {
			sum += std::exp(num - maxelm);
		}
		for (auto num: batch) {
			row.push_back(std::exp(num-maxelm)/sum);
		}
		mat.push_back(row);
	}
	return mat;
}

std::vector<std::vector<double>> SoftMax::backward(std::vector<std::vector<double>>& x) {
	return x;
}

ReLU::ReLU() {}

std::vector<std::vector<double>> ReLU::forward(std::vector<std::vector<double>>& x) {
	this->lastdata = x;
	std::vector<std::vector<double>> mat;
	for (auto batch: x) {
		std::vector<double> row;
		for (auto num: batch) {
			if (num > 0.0f) {
				row.push_back(num);
			}else {
				row.push_back(0);
			}
		}
		mat.push_back(row);
	}
	return mat;
}

std::vector<std::vector<double>> ReLU::backward(std::vector<std::vector<double>>& x) {
    auto mat = x;
	for (std::size_t i = 0; i < x.size(); i++) {
		for (std::size_t j = 0 ; j < x[i].size(); j++) {
			if (this->lastdata[i][j] <= 0) {
				mat[i][j] = 0;
			}
		}
	}
	return mat;
}
