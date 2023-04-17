#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

enum ActivationType{
	sigmoid,
	linear,
	relu,
	softmax
};

class Activation{
protected:
	std::vector<std::vector<double>> lastdata;
public:
	virtual std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& x) = 0;
	virtual std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& x) = 0;
};

class Sigmoid : public Activation {
public:
	Sigmoid();

	std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& x);
	std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& x);
};

class Linear : public Activation {
public:
	Linear();

	std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& x);
	std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& x);
};

class SoftMax : public Activation {
public:
	SoftMax();

	std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& x);
	std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& x);
};

class ReLU : public Activation {
public:
	ReLU();

	std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& x);
	std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& x);
};
