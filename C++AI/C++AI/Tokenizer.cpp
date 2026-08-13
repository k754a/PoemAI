#include <iostream>
#include <fstream> //read the file
#include <sstream> //for our buffer
#include <vector>
#include <unordered_map>
#include <algorithm>
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
		//having issues with gernation, caused by uppercase words
		if (currentWord != "<NEWLINE>")
		{
			for (char& c : currentWord) //loop through each letter
			{
				c = (char)tolower((unsigned char)c); //set c to lower
			}
		}

		wordCount[currentWord]++; //add to the current word
	}


	//ok, now we need to handle the ID of each word
	//we first gotta move our wordcount to a vector, so we can sort by rank
	//seraching online, this is the best way to do so, we say its a vector, its in pairs, (then state the pairs)
	//then we make our vector, and map each element!
	std::vector<std::pair<std::string, int>> sortedWords(wordCount.begin(), wordCount.end());

	// use fast sort (got this online)
	std::sort(sortedWords.begin(), sortedWords.end(), [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) { return a.second > b.second; });

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