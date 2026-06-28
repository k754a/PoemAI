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

	return 0;
}
