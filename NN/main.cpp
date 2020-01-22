/*
Eli Howitt. 12/2/2019. 
Attempt to make template for very basic neural networks.
No time to program stuff /:
*/

#include<iostream>
#include<ctime>

#include<chrono>

#include "NN.h"

#define OLC_PGE_APPLICATION
#include "olc.h"

class ErrorGrapher : public olc::PixelGameEngine
{
public:
	ErrorGrapher()
	{
		sAppName = "Error grapher for neural network learning process";
	}
public:

	int numInputs;
	int numHiddenLayers;
	int numOutputs;

	int* hiddenLayers;
	Network *logiNet;

	TrainingDataSet* trainingSet;

	int numCurrentTrainings;
	int numTargetTrainings;

	int numCheckUps;
	int checkUpInterval;

	int binaryResult;

	bool initScreen;

	std::chrono::system_clock::time_point startTime, currTime;
	std::chrono::duration<float> timeDifference;

	bool OnUserCreate() override
	{
		numInputs = 3;
		numHiddenLayers = 2;
		numOutputs = 8;

		hiddenLayers = new int[numHiddenLayers];
		hiddenLayers[0] = 9, hiddenLayers[1] = 6;//, hiddenLayers[2] = 6;
		logiNet = new Network(numInputs, hiddenLayers, numHiddenLayers, numOutputs);

		trainingSet = new TrainingDataSet(numInputs, numOutputs);
		trainingSet->momentum = 0.05f;

		numCurrentTrainings = 1;
		numTargetTrainings = 100000;

		checkUpInterval = 100;
		numCheckUps = numTargetTrainings / checkUpInterval;

		startTime = std::chrono::system_clock::now();
		currTime = startTime;

		initScreen = false;

		return true;
	}
	bool OnUserUpdate(float fElapsedTime) override
	{
		if (!initScreen)
		{
			for (int x = 0; x < ScreenWidth(); x++)
				for (int y = 0; y < ScreenHeight(); y++)
					Draw(x, y, olc::BLACK);

			initScreen = true;
		}
		else
		{
			if (numCurrentTrainings <= numTargetTrainings)
			{
				binaryResult = 0;
				for (int i = 0; i < numInputs; ++i)
				{
					int bit = rand() % 2;
					trainingSet->inputData[i] = bit;
					binaryResult += bit * pow(2, i);
				}

				for (int i = 0; i < numOutputs; ++i)
				{
					if (i == binaryResult)
						trainingSet->targetOutput[i] = 1;
					else
						trainingSet->targetOutput[i] = 0;
				}

				float ratioOfTrainingDone = (float)numCurrentTrainings / (float)numTargetTrainings;

				trainingSet->learningSpeed = 0.5 - (0.4f)*ratioOfTrainingDone;

				logiNet->learn(*trainingSet);

				if (numCurrentTrainings % checkUpInterval == 0)
				{
					std::cout << "Attempt number: " << numCurrentTrainings << " of " << numTargetTrainings << " - " << (int)(100 * (float)numCurrentTrainings / (float)numTargetTrainings) << "%.\n";

					std::cout << "Input: (";
					for (int i = 0; i < numInputs - 1; ++i)
						std::cout << trainingSet->inputData[i] << ", ";
					std::cout << trainingSet->inputData[numInputs - 1] << ").\n";
					
					std::cout << "Target result: (";
					for (int i = 0; i < numOutputs - 1; ++i)
						std::cout << trainingSet->targetOutput[i] << ", ";
					std::cout << trainingSet->targetOutput[numOutputs - 1] << ").";

					std::cout << "Network result: (";
					for (int i = 0; i < numOutputs - 1; ++i)
						std::cout << trainingSet->networkOutput[i] << ", ";
					std::cout << trainingSet->networkOutput[numOutputs - 1] << ").";

					std::cout << "Error: " << trainingSet->networkError << ".\n";

					currTime = std::chrono::system_clock::now();
					timeDifference = std::chrono::duration_cast<std::chrono::duration<float>>(currTime - startTime);

					std::cout << "Total elapsed time: " << timeDifference.count() << '\n';

					int numCurrentCheckUp = ((float)numCurrentTrainings / (float)numTargetTrainings) * numCheckUps;

					float error = trainingSet->networkError;

					int x = ((float)ScreenWidth() / (float)numCheckUps) * numCurrentCheckUp;
					int y = ScreenHeight() - (float)ScreenHeight() * (float)error;


					float redDist = error, greenDist = (1 - error), blueDist = abs(0.5 - error) * 2;
					olc::Pixel colour(((int)(redDist * 255)) % 256, ((int)(greenDist * 255)) % 256, ((int)(blueDist * 255)) % 256);

					DrawLine(x, y, x, ScreenHeight(), colour);

				}

				numCurrentTrainings++;
			}
		}
		return true;
	}
};
int main()
{
	srand(time(0));

	ErrorGrapher demo;
	//if (demo.Construct(256, 220, 4, 4))
	if (demo.Construct(900, 600, 1, 1))
		demo.Start();
	return 0;
}
