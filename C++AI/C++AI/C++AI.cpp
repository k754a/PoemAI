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
	}


	return 0;
}
