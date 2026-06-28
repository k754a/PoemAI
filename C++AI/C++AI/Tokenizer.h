#pragma once
#ifndef  Tokenizer_H
#define Tokenizer_H

#include <vector>
#include <unordered_map>
#include <string>


//our vars we want to get in our C++AI
extern std::vector<int> numTokens;

extern std::unordered_map<std::string, int> wordID;
extern std::unordered_map<int, std::string> vocab;


int token();

#endif