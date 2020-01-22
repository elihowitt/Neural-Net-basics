#include "NN.h"

Network::Network(const int & numInputs_,
	const int * hiddenLayers_, const int & numHiddenLayers_, const int & numOutputs_) :
	numInputs(numInputs_), numHiddenLayers(numHiddenLayers_), numOutputs(numOutputs_)
{
	numLayers = 1 + numHiddenLayers + 1;

	hiddenLayers = new int[numHiddenLayers];
	for (int i = 0; i < numHiddenLayers; ++i)
		hiddenLayers[i] = hiddenLayers_[i];

	layers = new Layer*[numLayers];

	//Input layer.
	//layers[0] = new Layer(numInputs, 1, leakyReLu, leakyReLuDerivative);
	layers[0] = new Layer(numInputs, 1, tanhf, tanhfDerivative);

	//hidden layers.
	for (int i = 0; i < numHiddenLayers; ++i)
		layers[1 + i] = new Layer(hiddenLayers[i], layers[i]->numNeurons, tanhf, tanhfDerivative);
		//layers[1 + i] = new Layer(hiddenLayers[i], layers[i]->numNeurons, leakyReLu, leakyReLuDerivative);

	//Output layer.
	layers[numLayers - 1] = new Layer(numOutputs, layers[numLayers - 2]->numNeurons, tanhf, tanhfDerivative);
	//layers[numLayers - 1] = new Layer(numOutputs, layers[numLayers - 2]->numNeurons, leakyReLu, leakyReLuDerivative);
}

Network::~Network()
{
	delete hiddenLayers;

	for (int i = 0; i < numLayers; ++i)
		delete layers[i];
	delete[] layers;
}

//void Network::learn(float * inputs, float * targetOutputs, float * results)
void Network::learn(TrainingDataSet& trainingSet_)
{
	//Input: 
	layers[0]->inputData(trainingSet_.inputData);

	//Feed forward:
	for (int i = 1; i < numLayers; ++i)
		layers[i]->process(layers[i - 1]);

	for (int i = 0; i < numOutputs; ++i)
		trainingSet_.networkOutput[i] = layers[numLayers - 1]->neurons[i]->output;

	//Backprop: 
	float currentTrainingError = 0;
	for (int i = 0; i < numOutputs; ++i)
		currentTrainingError += 
		(trainingSet_.networkOutput[i] - trainingSet_.targetOutput[i])*
		(trainingSet_.networkOutput[i] - trainingSet_.targetOutput[i]);

	//Average error:
	trainingSet_.networkError = sqrtf((float)currentTrainingError / (float)numOutputs);

	//Find personal error for each neuron:

	//Output layer:
	for (int i = 0; i < numOutputs; ++i)
	{
		Neuron* n = layers[numLayers - 1]->neurons[i];
		float delta = trainingSet_.targetOutput[i] - n->output;
		n->currentGradient = delta * n->activationFunctionDerivative((n->output));
	}
	//Hidden layers:
	int lastHiddenLayerIndex = (numLayers - 2);
	for (int i = lastHiddenLayerIndex; i >= 0; --i)
	{
		Layer* currLayer = layers[i], *nextLayer = layers[i + 1];
		for (int j = 0; j < currLayer->numNeurons; ++j)
		{
			Neuron* currNeuron = currLayer->neurons[j];

			float nextLayerAffect = 0.f;
			for (int k = 0; k < nextLayer->numNeurons; ++k)
			{
				Neuron* n = nextLayer->neurons[k];
				nextLayerAffect += n->currentGradient * n->prevLayerWeights[j];
			}

			currNeuron->currentGradient = nextLayerAffect * currNeuron->activationFunctionDerivative((currNeuron->output));
		}
	}

	//Now agian from output to input, update weights:
	for (int layerIndex = numLayers - 1; layerIndex > 0; --layerIndex)
	{
		Layer* currLayer = layers[layerIndex], *prevLayer = layers[layerIndex - 1];

		//Loop through each neuron:
		for (int neuronIndex = 0; neuronIndex < currLayer->numNeurons; ++neuronIndex)
		{
			Neuron* currNeuron = currLayer->neurons[neuronIndex];
			//Loop through and update weights:
			for (int weightIndex = 0; weightIndex < currNeuron->prevLayerSize; ++weightIndex)
			{
				Neuron* prevNeuron = prevLayer->neurons[weightIndex];

				float currDelta = trainingSet_.learningSpeed * prevNeuron->output * currNeuron->currentGradient +
					trainingSet_.momentum * (currNeuron->previouseDeltas)[weightIndex];

				currNeuron->prevLayerWeights[weightIndex] += currDelta;
				(currNeuron->previouseDeltas)[weightIndex] = currDelta;
					
			}

			float biasDelta = trainingSet_.learningSpeed * (1) * currNeuron->currentGradient +
				trainingSet_.momentum * currNeuron->prevBiasDelta;
			currNeuron->biasWeight += biasDelta;
			currNeuron->prevBiasDelta = biasDelta;
		}


	}

}

Layer::Layer(const int & numNeurons_, const int& prevLayerSize_,
	float(*activationFunction_)(float), float(*activationFunctionDerivative_)(float)): numNeurons(numNeurons_)
{
	bias = 1.f;// (float)rand() / (float)RAND_MAX;

	neurons = new Neuron*[numNeurons];

	for (int i = 0; i < numNeurons; ++i)
		neurons[i] = new Neuron(prevLayerSize_, bias, activationFunction_, activationFunctionDerivative_);
}

Layer::~Layer()
{
	for (int i = 0;i < numNeurons; ++i)
		delete neurons[i];
	delete[] neurons;
}

void Layer::process(Layer * prev_)
{
	float* prevValues = new float[prev_->numNeurons];
	for (int i = 0; i < prev_->numNeurons; ++i)
		prevValues[i] = prev_->neurons[i]->output;

	for (int i = 0; i < numNeurons; ++i)
		neurons[i]->calculateValue(prevValues);

	delete[] prevValues;
}

void Layer::inputData(float * data_)
{
	for (int i = 0; i < numNeurons; ++i)
		neurons[i]->output = data_[i];
}

Neuron::Neuron(const int & prevLayerSize_, const float& bias_,
	float(*activationFunction_)(float), float(*activationFunctionDerivative_)(float)) :
	prevLayerSize(prevLayerSize_), 
	activationFunction(activationFunction_), activationFunctionDerivative(activationFunctionDerivative_)
{
	prevLayerWeights = new float[prevLayerSize];
	previouseDeltas = new float[prevLayerSize];

	for (int i = 0; i < prevLayerSize; ++i)
	{
		//Network initialization:		
		prevLayerWeights[i] = (float)rand() / (float)RAND_MAX;
		previouseDeltas[i] = 0;
	}

	biasWeight = (float)rand() / (float)RAND_MAX;
	prevBiasDelta = 0;
}

Neuron::~Neuron()
{
	delete[] prevLayerWeights;
	delete[] previouseDeltas;
}

void Neuron::calculateValue(float * prevLayerValues_)
{
	float sum = 0;

	for (int i = 0; i < prevLayerSize; ++i)
		sum += prevLayerValues_[i] * prevLayerWeights[i];


	output = activationFunction((sum + 1*biasWeight));
}

float sigmoid(float input_)
{
	return (float)1 / (float)(1 + std::exp((-1)*input_));
}

float sigmoidDerivative(float input_)
{
	return input_ * (1 - input_);
	//return sigmoid(input_)*(1 - sigmoid(input_));
}

float tanhfDerivative(float input_)
{
	return 1 - input_ * input_;
}

float reLu(float input_)
{
	if (input_ > 0)
		return input_;
	return 0.f;
}

float reLuDerivative(float input_)
{
	if (input_ > 0)
		return 1;
	return 0;
}

float leakyReLu(float input_)
{
	if (input_ > 0)
	{
		if (input_ < 1)
			return input_;
		else
			return 1 + (input_ - 1)*LEAKY_CONSTANT;
	}
	return input_ * LEAKY_CONSTANT;
}

float leakyReLuDerivative(float input_)
{
	if (input_ > 0)
	{
		if (input_ < 1)
			return 1;
		else
			return LEAKY_CONSTANT;
	}
	return LEAKY_CONSTANT;
}

TrainingDataSet::TrainingDataSet(
	const int & numInputs_, const int & numOutputs_):
	numInputs(numInputs_), numOutputs(numOutputs_)
{
	inputData = new float[numInputs];
	targetOutput = new float[numOutputs];
	networkOutput = new float[numOutputs];
}


TrainingDataSet::~TrainingDataSet()
{
	delete[] inputData;
	delete[] targetOutput;
	delete[] networkOutput;
}
