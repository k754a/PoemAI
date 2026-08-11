//Ok, this is the c++ version of the python code, i'm much more comfortable in c++ than python overall!
//first, we need to handle our data overall.

//We need to remove and simplify things like punctuation and numbers
#include "ProcessData.h"
#include "Tokenizer.h"

#include <cmath> //need this for the softmax
#include <fstream> 
#include <iostream>
#include <cstdlib>

int main() {
	
	handleData();

	std::cout << "Starting Token Processing! " << std::endl;
	token();

	//grab the token num here!
	std::vector<int> numberOfTokens = numTokens;

	//Ok, now we need context, and how much it should have
	//LLMS have about 1m+ but that takes a long time to train, however, because c++ is faster, we can do a context window of 15-30!

	int context = 12; //context 64 -> 12 to impove pattern reco


	//what we are doing is spliting our text up into 4 chars, and through taking our whole txt file, we can start to get better at guessing
	//and understanding patterns!

	//ok now we need to make the Inputs, and targets
	//the inputs is the 3 context words we give it
	//we have a vector in a vector because each flashcard has a list of ids
	std::vector<std::vector<int>> Inputs;
	std::vector<int> Targets; //this holds our guess/target! we use it to train the ai!
	
	//ok this is our sliding window loop
	//this lets us assign flashcards

	//we make sure that the num of tokens is an int, or we will have issues
	for (int i = 0; i < (int)numberOfTokens.size() - context; i++)
	{
		//grab the context words, starting at i!
		std::vector<int> currentInts;
		for (int j = 0; j < context; j++)
		{
			currentInts.push_back(numberOfTokens[i + j]);
		}

		//ok we got the current inputs, now we need the guess
		//we get it by finding the word right after the window!
		int currentGuess = numberOfTokens[i + context];

		//then we save it
		Inputs.push_back(currentInts);
		Targets.push_back(currentGuess);
	}

	std::cout << "Flashcard count: " << Inputs.size() << std::endl;

	//Now we need to fill our 2d matrix, and fill it with random data

	//we make our word count

	int wordCount = vocab.size() + 1; //we do + 1 to prevent errors
	int neuronCount = 512; //how large each word vector is!

	//the python version, i had 2 diffrent loops, c++ i can just use one!
	//we need to fill the vectors with random floats between 0-1
	std::vector<std::vector<float>> weights(wordCount, std::vector<float>(neuronCount));//we make a [, []], so we can store each val
	//do the same again
	std::vector<std::vector<float>> outputWeights(wordCount, std::vector<float>(neuronCount));

	//ok, for random values in c++, i did some searching, and we use a srand and put the system time in as the seed! (as because it is always going up, its always random)
	srand(time(0));

	//now lets fill the values in

	//for each word in the word count
	for (int w = 0; w < wordCount; w++)
	{
		//ok we loop through each "neuron"
		for (int n = 0; n < neuronCount; n++)
		{
			//ok, assign them, both w + n

			//ok, apparently, i did some reading, and i didnt center the weights! if i dont, we get crazy errors

			weights[w][n] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
			outputWeights[w][n] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
		}

	}

	//a little bit of DEBUG, just so i know if things worked or not
	std::cout << "Weights created! " << std::endl;



	//ok, its training time!

	//make the learning rate (how much our weights get adjusted based on changes
	float learningRate = 0.01f;

	//handle epochs, how many times we go through the data!
	int epochs = 50;

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		//say the epoch we are on
		std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
		//ok, we loop through all our inputs, adjusting each part as we go along

		//first we adjust the learning rate, as i found out doing some research, it ends up unlearning things!

		float epochLearningRate = learningRate / (1.0f + epoch * 0.1f);

		for (int all = 0; all < (int)Inputs.size(); all++)
		{
			//reset the hiddenState to 0's, as we want to start it fresh for each flash card
			//we can make a more simple list, and its so much faster
			float hiddenState[512] = {};

			//for each ID in our context window, grab its weights row
			//the add it into the hidden state, this builds a summary of the context
			for (int w = 0; w < (int)Inputs[all].size(); w++)
			{
				int wordId = Inputs[all][w]; //get the ID number of the word, this is kinda confusing, as we have another WordID
				if (wordId <= 0 || wordId >= wordCount) continue; //skip if we meet these conditions, and this prevents erros
				//add this to our row, for each neuron
				//i forgot to change < so it would have been forever lol
				for (int n = 0; n < neuronCount; n++)
				{
					hiddenState[n] += weights[wordId][n]; //update the hidden state
				}
			}
			for (int n = 0; n < neuronCount; n++)
			{
				hiddenState[n] /= (float)context;
				if (hiddenState[n] > 1.0f) hiddenState[n] = 1.0f;
				else if (hiddenState[n] < -1.0f) hiddenState[n] = -1.0f;
			}

			//ok, now we need to make a guess of what word comes next
			//we do that by scoring every word in our vocab
			//and if our word is off, we adjust our guess, and try again
			//doing this over and over, we are able to get good results over lots of training!

			//ok, now we need to update the weights a bit
			//this teaches the weights matrix what looks right
			//we use a small learning rate, to change cleanly!


			//now that we have done everything, we are onto the final step
			//reward / punish, we do this if the ai makes a bad guess, and so that
			//next guess the incorrect word scores lower, this lets us train the ai by teaching it whats right or wrong!

			int targetId = Targets[all]; //grab our targets, for the current part
			if (targetId <= 0 || targetId >= wordCount) continue; //check to make sure we arnt going to pass bad data in

			float inputGrad[512] = {}; //this array will track the weights based on the ouput
			for (int out = 0; out < neuronCount; out++)
			{
				//we increase the signficance of the word in this patter, if its correct
				outputWeights[targetId][out] += epochLearningRate * hiddenState[out];

				//clamp
				if (outputWeights[targetId][out] > 1.0f) outputWeights[targetId][out] = 1.0f;
				if (outputWeights[targetId][out] < -1.0f) outputWeights[targetId][out] = -1.0f;

				inputGrad[out] += outputWeights[targetId][out];
			}




			//ok we still punish bad guesses, however, we dont do this every step cause its super slow
			//so, im going to do it every 10000ish guess, as that still teaches it, but saves us time
			int NegativeSamplesNum = 15; //punish 15 words

			
			for (int i = 0; i < NegativeSamplesNum; i++)
			{
				//skip hte 0 id, and make sure we dont punish the correct word
				int badGuessId;
				do {
					badGuessId = 1 + rand() % (wordCount - 1); //skip the first id to get the ai to run better
				} while (badGuessId == targetId);

				for (int out = 0; out < neuronCount; out++) {
					// We subtract here to punish!
					outputWeights[badGuessId][out] -= epochLearningRate * hiddenState[out];

					//clamp
					if (outputWeights[badGuessId][out] > 1.0f) outputWeights[badGuessId][out] = 1.0f;
					if (outputWeights[badGuessId][out] < -1.0f) outputWeights[badGuessId][out] = -1.0f;

					inputGrad[out] -= outputWeights[badGuessId][out];

				}
			}
			
			//simple back propo, to update the input weights
			for (int w = 0; w < (int)Inputs[all].size(); w++) {
				int wordId = Inputs[all][w];
				if (wordId <= 0 || wordId >= wordCount) continue; //check to make sure we arnt going to pass bad data in

				for (int n = 0; n < neuronCount; n++) {
					weights[wordId][n] += epochLearningRate * 1.0f * inputGrad[n];
					//check if the word maches, and this will prevent issues wiht everything being throw together, and only chosing 15 different words

					if (weights[wordId][n] > 1.0f) weights[wordId][n] = 1.0f;
					if (weights[wordId][n] < -1.0f) weights[wordId][n] = -1.0f;

				}
			}
			


			//ok now we want to print progress, so we know its working, but std every loop is so slow
			//so we % by 5000, and if its 0, that means its been 5000

			//ok, so we also handle our last loop, if the inputs are == to the all!
			if (all % 5000 == 0 || all == (int)Inputs.size() - 1)
			{
				std::cout << "Progress: " << all << "/" << Inputs.size() << std::endl;
			}


		}

	}

		
	

	//ok we are gonna save the model! however we need to rewrite a few things, as doing .json sucks
	//so im just gonna save it to a txt file

	std::ofstream modelFile("poem_model.txt");

	//check to make sure its open
	if (!modelFile.is_open())
	{
		std::cout << "ERROR - Could not save model" << std::endl;
	}

	modelFile.precision(9); //cut the float per to 9 

	//first we save the word + neuron count
	modelFile << wordCount << "\n";
	modelFile << neuronCount << "\n";

	//then we save the vocab size
	modelFile << vocab.size() << "\n"; //we make a new line after saying how many words there are!


	//then we should write out each word and its ID pairs

	//for each loop 
	for (auto& pair : vocab)
	{
		modelFile << pair.first << " " << pair.second << "\n";
	}


	//save input and output weights, and there neurons
	for (int w = 0; w < wordCount; w++)
	{
		for (int n = 0; n < neuronCount; n++)
		{
			//write the values
			modelFile << weights[w][n] << " ";
		}
		modelFile << "\n"; 
	}

	//the exact same thing for our output weights!
	for (int w = 0; w < wordCount; w++)
	{
		for (int n = 0; n < neuronCount; n++)
		{
			modelFile << outputWeights[w][n] << " ";
		}
		modelFile << "\n";
	}

	modelFile.close(); //cleanup

	std::cout << "Done!" << std::endl;



	return 0;
}
