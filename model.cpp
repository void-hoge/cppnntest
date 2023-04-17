#include "model.hpp"

#include <cmath>
#include <iostream>

Model::Model(std::size_t inputsize):
	inputsize(inputsize),
	outputsize(inputsize),
	m_loss("cen") {}

void Model::add_dense_layer(std::size_t unit, ActivationType act) {
	this->model.push_back(Layer::Dense(this->outputsize, unit, act));
	this->outputsize = unit;
}

std::vector<std::vector<double>> Model::predict(std::vector<std::vector<double>>& data) {
	auto dst = data;
	for (auto& layer: this->model) {
		dst = layer.forward(dst);
	}
	return dst;
}

double Model::mean_squared_error(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& t) {
	double res = 0;
	for (std::size_t i = 0; i < y.size(); i++) {
		double sum = 0;
		for (std::size_t j = 0; j < y[i].size(); j++) {
			sum += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
		}
		sum /= 2*y.size();
		res += sum;
	}
	return res;
}

std::vector<std::vector<double>> Model::mean_squared_error_back(std::vector<std::vector<double>>& absolute_error) {
	return absolute_error;
	std::vector<std::vector<double>> mat;
	for (std::size_t i = 0; i < absolute_error.size(); i++) {
		std::vector<double> row;
		for (std::size_t j = 0; j < absolute_error[i].size(); j++) {
			row.push_back(2*std::abs(absolute_error[i][j]));
		}
		mat.push_back(row);
	}
	return mat;
}

double Model::mean_cross_entropy_error(std::vector<std::vector<double>>& y, std::vector<std::vector<double>>& t) {
	double res = 0;
	for (std::size_t i = 0; i < y.size(); i++) {
		double sum = 0;
		double delta = 1e-7;
		for (std::size_t j = 0; j < y[i].size(); j++) {
			sum += t[i][j] * std::log(y[i][j] + delta);
		}
		sum *= -1;
		sum /= y.size();
		res += sum;
	}
	return res;
}

std::vector<std::vector<double>> Model::mean_cross_entropy_error_back(std::vector<std::vector<double>>& error) {
	return error;
}

double Model::caluculate_loss(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y) {
	std::vector<std::vector<double>> t = this->predict(x);
	if (this->m_loss == "cen") {
		return this->mean_cross_entropy_error(t, y);
	}else {
		return this->mean_squared_error(t, y);
	}
}

std::vector<std::vector<double>> Model::numerical_gradient_weight(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, std::size_t idx) {
	double h = 1e-4;
	std::vector<std::vector<double>> grad(this->model[idx].weights.size(), std::vector<double>(this->model[idx].weights[0].size()));

	for (std::size_t i = 0; i < grad.size(); i++) {
		for (std::size_t j = 0; j < grad[i].size(); j++) {
			double tmp = this->model[idx].weights[i][j];

			this->model[idx].weights[i][j] = tmp + h;
			double fxh1 = this->caluculate_loss(x, y);

			this->model[idx].weights[i][j] = tmp - h;
			double fxh2 = this->caluculate_loss(x, y);

			grad[i][j] = (fxh1-fxh2) / (2*h);
			this->model[idx].weights[i][j] = tmp;
		}
	}
	return grad;
}

std::vector<double> Model::numerical_gradient_bias(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, std::size_t idx) {
	double h = 1e-4;
	std::vector<double> grad(model[idx].biases.size());
	for (std::size_t i = 0; i < grad.size(); i++) {
		double tmp = this->model[idx].biases[i];

		this->model[idx].biases[i] = tmp + h;
		double fxh1 = this->caluculate_loss(x, y);

		this->model[idx].biases[i] = tmp - h;
		double fxh2 = this->caluculate_loss(x, y);

		grad[i] = (fxh1-fxh2) / (2*h);
		this->model[idx].biases[i] = tmp;
	}
	return grad;
}

void Model::backward() {
	std::vector<std::vector<double>> data;
	if (this->m_loss == "mse") {
		data = this->mean_squared_error_back(this->diff_error);
	}else {
		data = this->mean_cross_entropy_error_back(this->diff_error);
	}

	for (auto iter = this->model.rbegin(); iter != this->model.rend(); iter++) {
		data = iter->backward(data);
	}
}

std::vector<double> Model::fit(
	std::size_t step, double learningrate,
	std::vector<std::vector<double>>& x,
	std::vector<std::vector<double>>& y,
	std::size_t batchsize, std::string loss) {

	this->m_loss = loss;
	std::vector<double> history;
	for (std::size_t currentstep = 0; currentstep < step; currentstep++) {
		std::vector<std::vector<double>> batchx, batchy;
		for (std::size_t i = 0; i < batchsize; i++) {
			batchx.push_back(x[(batchsize * currentstep + i) % x.size()]);
			batchy.push_back(y[(batchsize * currentstep + i) % y.size()]);
		}

		for (std::size_t idx = 0; auto& layer: this->model) {
			std::vector<std::vector<double>> weight_grad = this->numerical_gradient_weight(batchx, batchy, idx);
			std::vector<double> bias_grad = this->numerical_gradient_bias(batchx, batchy, idx);

			for (std::size_t i = 0; i < weight_grad.size(); i++) {
				for (std::size_t j = 0; j < weight_grad[i].size(); j++) {
					layer.weights[i][j] -= learningrate * weight_grad[i][j];
				}
			}

			for (std::size_t i = 0; i < bias_grad.size(); i++) {
				layer.biases[i] -= learningrate * bias_grad[i];
			}

			idx++;
		}
		history.push_back(this->caluculate_loss(batchx, batchy));
	}
	return history;
}

std::vector<double> Model::fitbp(
	std::size_t step, double learningrate,
	std::vector<std::vector<double>>& x,
	std::vector<std::vector<double>>& y,
	std::size_t batchsize, std::string loss) {
	std::vector<double> history;
	this->m_loss = loss;
	for (std::size_t current_step = 0; current_step < step; current_step++) {
		std::vector<std::vector<double>> batch_x, batch_y;
		diff_error.clear();
		for (std::size_t i = 0; i < batchsize; i++) {
			batch_x.push_back(x[(batchsize*current_step+i)%x.size()]);
			batch_y.push_back(y[(batchsize*current_step+i)%y.size()]);
		}
		auto test_y = this->predict(batch_x);
		for (std::size_t i = 0; i < test_y.size(); i++) {
			std::vector<double> t;
			for (std::size_t j = 0; j < test_y[i].size(); j++) {
				t.push_back((test_y[i][j] - batch_y[i][j]) / batchsize);
			}
			this->diff_error.push_back(t);
		}
		this->backward();
		double loss_step;
		if (this->m_loss == "mse") {
			loss_step = mean_squared_error(test_y, batch_y);
		}else {
			loss_step = mean_cross_entropy_error(test_y, batch_y);
		}

		for (auto &layer: this->model) {
			std::vector<std::vector<double>> grad_weights = layer.grad_weights;
			std::vector<double> grad_biases = layer.grad_biases;
			for (std::size_t i = 0; i < grad_weights.size(); i++) {
				for (std::size_t j = 0; j < grad_weights[i].size(); j++) {
					layer.weights[i][j] -= learningrate * grad_weights[i][j];
				}
			}
			for (std::size_t i = 0; i < grad_biases.size(); i++) {
				layer.biases[i] -= learningrate * grad_biases[i];
			}
		}
		history.push_back(loss_step);
		std::cout << "current_step: " << current_step+1 << "/" << step << ", loss: " << loss_step << std::endl;
	}
	return history;
}
