#include <iostream>
#include <fstream> //read the file
#include <sstream> //for our buffer
#include <vector>
#include <unordered_map>

#include "Tokenizer.h"


std::string poemText; //we dont want this accessable

//we want these accessable 

std::vector<int> numTokens; //global, this is for our []
std::unordered_map<std::string, int> wordID; //global, for our wordID {}
std::unordered_map<int, std::string> vocab; //global, for our vocab {}


int token()
{

	//load the txt file
	std::ifstream poem("poems.txt");

	//then we make sure it opened
	if (!poem.is_open())
	{
		std::cout << "ERROR - COULD NOT LOAD FILE TO TOKENIZE" << std::endl;
		return -1;
	}

	//we do the same thing before
	std::stringstream buffer;

	buffer << poem.rdbuf();

	//then send it to a string

	poemText = buffer.str();

	//ok, now that we have loaded the full text, lets split it into words
	std::unordered_map<std::string, int> wordCount; //its a dictonary

	//we can do this a bit better than the python version!, as we can use stringstream

	//load the string stream
	std::stringstream ss(poemText);
	std::string currentWord;

	//super cool thing we can do! we can just see the word, and add 1, c++ already checks and finds it!
	while (ss >> currentWord)
	{
		wordCount[currentWord]++; //add to the current word
	}


	//ok, now we need to handle the ID of each word
	//we first gotta move our wordcount to a vector, so we can sort by rank
	//seraching online, this is the best way to do so, we say its a vector, its in pairs, (then state the pairs)
	//then we make our vector, and map each element!
	std::vector<std::pair<std::string, int>> sortedWords(wordCount.begin(), wordCount.end());

	//ok now we sort, im gonna use bubble sort for this cause tis easy!
	for (int s = 0; s < sortedWords.size() - 1; s++)
	{
		//each pass we move the next largest val towards the front

		//so for example we start 1, 2, 3, 4
		//we compare  1 and 4, then 2 and 3, then swap!
		for (int i = 0; i < sortedWords.size() - s - 1; i++)
		{
			//compare the second value of each element, and if one is bigger than another, swap.
			if (sortedWords[i].second < sortedWords[i + 1].second)
			{
				//swap the 2 elements
				std::swap(sortedWords[i], sortedWords[i + 1]);
			}
		}

	}

	//ok, now we assign the id's
	int currentID = 1;

	for (int i = 0; i < sortedWords.size(); i++)
	{
		std::string word = sortedWords[i].first; //first is our "string"

		//save the ID to the word
		vocab[currentID] = word;

		//save to the wordID
		wordID[word] = currentID;

		//then we go to the next one
		currentID++;
	}


	//finaly we have our num tokens!
	std::stringstream tokenStream(poemText); //we need to make a new one, cause we alr made it to the end of the last one!

	//look through each word, and conver the word into the id!
	while (tokenStream >> currentWord)
	{
		//we add to our tokens, by getting the word id out of the current word!
		numTokens.push_back(wordID[currentWord]);
	}



	std::cout << "Done!" << std::endl;

	return 0;
}