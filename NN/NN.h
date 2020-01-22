#pragma once

#include<random>
#include<cmath>

float sigmoid(float input_);
float sigmoidDerivative(float input_);

float tanhfDerivative(float input_);

float reLu(float input_);
float reLuDerivative(float input_);

const float LEAKY_CONSTANT = 0.005f;

float leakyReLu(float input_);
float leakyReLuDerivative(float input_);

class TrainingDataSet
{
public:
	TrainingDataSet(const int& numInputs_, const int& numOutputs_);
	~TrainingDataSet();

	int numInputs;
	int numOutputs;

	float* inputData;
	float* targetOutput;

	//The actuall output.
	float* networkOutput;
	//Resulting error.
	float networkError;

	float learningSpeed;
	float momentum;
};

class Neuron
{
public:
	Neuron(const int& prevLayerSize_, const float& bias_, 
		float(*activationFunction_)(float), float(*activationFunctionDerivative_)(float));
	~Neuron();

	void calculateValue(float* prevLayerValues_);

	int prevLayerSize;

	float output;

	float* prevLayerWeights;
	float* previouseDeltas;//Array of the previous delta for each weight.

	float biasWeight;
	float prevBiasDelta;

	float currentGradient;

	float(*activationFunction)(float);
	float(*activationFunctionDerivative)(float);
};

class Layer
{
public:
	Layer(const int& numNeurons_, const int& prevLayerSize_,
		float(*activationFunction_)(float), float(*activationFunctionDerivative_)(float));
	~Layer();

	//Given the previouse layer (assuming its allready been processed), calculate the neurons values.
	void process(Layer* prev_);

	//In case this layer is the input layer we can directly set the neurons values.
	void inputData(float* data_);
	
	float bias;

	int numNeurons;

	Neuron** neurons;

};

class Network
{
public:
	Network(const int& numInputs_, const int* hiddenLayers_,
		const int& numHiddenLayers_, const int& numOutputs_);
	~Network();

	/*
		-Feed inputs.
		-Propagate data through hidden layers.
		-Calculate finale results and uptade 'results' param.
		-Compare results with 'targetOutputs' and find error value.
		-Backpropagate and tweak network values (weights (and biasies?)).
	*/
	//void learn(float* inputs, float* targetOutputs, float* results);
	void learn(TrainingDataSet& trianingSet_);

private:

	int numInputs; //How many input nodes.

	int numHiddenLayers;//How many hidden layers.
	int* hiddenLayers;//How many nodes in each hidden layer.

	int numOutputs;//How many output nodes.

	int numLayers;//In total.

	Layer ** layers; 

};


