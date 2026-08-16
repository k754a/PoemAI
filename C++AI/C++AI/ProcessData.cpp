#include "ProcessData.h"
#include <fstream> //for loading files
#include <iostream> //for printing text, and calling some functs
#include <string> //for handling strings
#include <sstream> //for our buffer

//this funct takes our data, and handles it!
//its hard to handle something like a csv, so we are just gonna handle the txt side
int handleData()
{

	//we need this for our write
	std::string poemText;



	//ok open the file!
	std::fstream poem("poems.txt");

	//then check we got it
	if(!poem.is_open())
	{ 
		std::cout << "ERROR, COULD NOT OPEN FILE " << std::endl;
		return -1;
	}
	else {
		//ok we made sure that its open!, so we are gonna parse our file
		//what the plan is, we have a var this is = to the poem val, then we get the amount of chars
		//then go through each char, compare to our list of unwanted chars, and remove them! 
		//Then write to the file
		//then save it!

		//we can load everything at once, using a buffer
		std::stringstream buffer;

		buffer << poem.rdbuf();

		//then send it to a string

		poemText = buffer.str();

		//make a list of our unwanted chars

		std::string remove_chars = "/\\.,:;'+=-|\"'!@#?[]{}$%^&*())1234567890";

		//loop backwards
		std::string cleanedText;
		cleanedText.reserve(poemText.size());
		for (char c : poemText)
		{
			//ok, now that we loop through each char, lets check
			//super easy, like my c++ parser!
			//we check to see if we can find a char, that is the same in remove_chars
			//then erase!
			if (remove_chars.find(c) == std::string::npos)
			{
				//ok, we have found the char
				//erase is slow, but idrc, as this will run once!
				cleanedText.push_back(c);
			}
		}
		poemText.swap(cleanedText);

		//teaches the ai new lines!
		size_t pos = 0;
		//we need to teach it <newline tags, so we find the \n, and if we find one, we replace this, we go through the list!
		//this is super easy, as this is normal in the browser engine!
		while ((pos = poemText.find('\n', pos)) != std::string::npos)
		{
			poemText.replace(pos, 1, " <NEWLINE> ");
			pos += 11; //after we change it, move past to avoid gliches
		}

		//close before we write
		poem.close();


		//ok we are done, lets write!
		std::ofstream outPoem("poems.txt");
		if (!outPoem.is_open())
		{
			std::cout << "ERROR, COULD NOT OPEN FILE " << std::endl;
			return -1;
		}

		//done!
		outPoem << poemText;

		outPoem.close();

		std::cout << "Done!" << std::endl;

	}
	

	return 0;
}