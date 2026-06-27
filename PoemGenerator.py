#ok, for this project, i have no idea what im doing, but the plan is building a poem generator.
#im thinking of building a simple tokenizer, and small llm from scratch to generate poems, with a start word
#that will allow me to avoid complex things like getting a huge ai model to guess words
#then once i get a few poems working, we are gonna do way more!

#ok first, we need to build a tokenizer
#the challange in this project is that im not using any external libaries, so i need to code everthing from scratch.
#What im gonna do first, is remove all punctuation and chars from the text, as that will help with tokenization, because im not gonna try to trian it punctuation too.

import ProcessData
import Tokenizer

ProcessData.HandleData() # on start, handle the data
Tokenizer.Token() # tokenize the data

numTokens = Tokenizer.numTokens #pass the value here

#we need to build a sliding window, first we need to tell the ai, how much context it should have
#large llms have abou 1m+ currently, but that takes a long time to train and stuff, so lets do 3 for now

context = 3 #tells the ai "read 3~ words to get the context"


#what we are doing is spliting our text up into 4 chars, and through taking our whole txt file, we can start to get better at guessing
#and understanding patterns!

#ok now we need to make the Inputs, and targets
#the inputs is the 3 context words we give it
Inputs = []
#the context is the word or words it must guess based off of whats most likey
Targets = []

#ok now we need a loop to loop through the list of numtokens, BUT we need to stop the list early, becasue if we go to the very last word, for example its "the end"
#what happens is we cant use it, as our first input is "end" and we dont have 2 others, or any targets, so we ignore the last 3
#so taking that, we loop through everything, but the last 3
for i in range(0, len(numTokens) - context):

    #first we need exactly 3 words for our input context
    currentInputs = numTokens[i : i + context] # we use : to take out a slice, starting at i then moving 3 up!

    currentGuess = numTokens[i + context] # just use the i+context

    #then we save to our flashcards
    
    Inputs.append(currentInputs)
    Targets.append(currentGuess)


#now that we are done, lets print how many we have
print("TOTAL: ", len(Inputs))