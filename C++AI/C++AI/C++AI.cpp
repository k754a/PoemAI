//Ok, this is the c++ version of the python code, i'm much more comfortable in c++ than python overall!
//first, we need to handle our data overall.

//We need to remove and simplify things like punctuation and numbers
#include "ProcessData.h"
#include "Tokenizer.h"



#include <iostream>
int main() {
	
	handleData();

	std::cout << "Starting Token Processing! " << std::endl;
	token();

	//grab the token num here!
	std::vector<int> numberOfTokens = numTokens;

	//Ok, now we need context, and how much it should have
	//LLMS have about 1m+ but that takes a long time to train, however, because c++ is faster, we can do a context window of 15-30!

	int context = 20; //context


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
	int neuronCount = 100; //how large each word vector is!

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
			weights[w][n] = (float)rand() / RAND_MAX; //doing randmax locks it between 0 or 1!

			outputWeights[w][n] = (float)rand() / RAND_MAX; //doing randmax locks it between 0 or 1!
		}

	}

	//a little bit of DEBUG, just so i know if things worked or not
	std::cout << "Weights created! " << std::endl;



	//ok, its training time!

	//make the learning rate (how much our weights get adjusted based on changes
	float learningRate = 0.01f;

	//ok, we loop through all our inputs, adjusting each part as we go along
	for (int all = 0; all < (int)Inputs.size(); all++)
	{
		//reset the hiddenState to 0's, as we want to start it fresh for each flash card
		std::vector<float> hiddenState(neuronCount, 0);

		//for each ID in our context window, grab its weights row
		//the add it into the hidden state, this builds a summary of the context
		for (int w = 0; w < (int)Inputs[all].size(); w++)
		{
			int wordId = Inputs[all][w]; //get the ID number of the word, this is kinda confusing, as we have another WordID

			//add this to our row, for each neuron
			//i forgot to change < so it would have been forever lol
			for (int n = 0; n < neuronCount; n++)
			{
				//ok, add it
				hiddenState[n] += weights[wordId][n];
			}
		}



		//ok, now we need to make a guess of what word comes next
		//we do that by scoring every word in our vocab
		//and if our word is off, we adjust our guess, and try again
		//doing this over and over, we are able to get good results over lots of training!

		float greatestScore = -INFINITY;
		int greatestScoreID = 0; //holds our greatest score!

		for (int i = 0; i < wordCount; i++) //loop through our word count!
		{
			float wordScore = 0;

			//we multiply 100 times, and then sum the up
			for (int out = 0; out < neuronCount; out++)
			{
				wordScore += hiddenState[out] * outputWeights[i][out];
			}

			//then we update the highest score, for whatever it is

			if (wordScore > greatestScore)
			{
				greatestScore = wordScore;
				greatestScoreID = i;
			}
		}

		//ok, now we need to update the weights a bit
		//this teaches the weights matrix what looks right
		//we use a small learning rate, to change cleanly!


		//note, i found out about this later, after the python change
		//i didnt do this before, and it would cause words to be very favorited and repeated
		for (int w = 0; w < (int)Inputs[all].size(); w++)
		{
			//ok we get the word ID
			int wordId = Inputs[all][w]; //like last time

			//ok so for each neuron, we update our weights
			for (int n = 0; n < neuronCount; n++)
			{
				weights[wordId][n] += learningRate * 0.1f;
			}
		}



		//now that we have done everything, we are onto the final step
		//reward / punish, we do this if the ai makes a bad guess, and so that
		//next guess the incorrect word scores lower, this lets us train the ai by teaching it whats right or wrong!

		int targetId = Targets[all]; //grab our targets, for the current part


		for (int out = 0; out < neuronCount; out++)
		{
			//we increase the signficance of the word in this patter, if its correct
			outputWeights[targetId][out] += learningRate;
		}

		//if its incorrect tho, we lower it
		if (greatestScoreID != targetId)
		{
			for (int out = 0; out < neuronCount; out++)
			{
				//same thing, but down
				outputWeights[greatestScoreID][out] -= learningRate;
			}
		}


		//ok now we want to print progress, so we know its working, but std every loop is so slow
		//so we % by 1000, and if its 0, that means its been 1000

		//ok, so we also handle our last loop, if the inputs are == to the all!
		if (all % 1000 == 0 || all == (int)Inputs.size() - 1)
		{
			std::cout << "Progress: " << all << "/" << Inputs.size() << std::endl;
		}




	}


	return 0;
}
