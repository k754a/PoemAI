#ok, for this project, i have no idea what im doing, but the plan is building a poem generator.
#im thinking of building a simple tokenizer, and small llm from scratch to generate poems, with a start word
#that will allow me to avoid complex things like getting a huge ai model to guess words
#then once i get a few poems working, we are gonna do way more!

#ok first, we need to build a tokenizer
#the challange in this project is that im not using any external libaries, so i need to code everthing from scratch.
#What im gonna do first, is remove all punctuation and chars from the text, as that will help with tokenization, because im not gonna try to trian it punctuation too.

import ProcessData
import Tokenizer
import random
import json

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


# embedding layer 


#ok, now we need to fill a 2d matrix, and fill it with random data.
#we know our data has exactly 2500 words (im def gonna add somthing to get the words normaly currently this is fine)
wordCout = wordCout = len(Tokenizer.vocab) + 1
#neuron count
neuronCount = 100

#list to store our weights
weights = []

#for each word 
for w in range(0, wordCout):
    #we make a new neuron list
    neuron = []

    #then we loop throught the 100 neurons for each word
    for n in range(0, neuronCount):

        #add something random
        neuron.append(random.random())

    #add to our weights
    weights.append(neuron)
   
print(weights)




outputWeights = []
learningRate = 0.01

#output weights will crash, so we do this: 
for w in range(0, wordCout):
    #we make a new neuron list
    neuron = []

    #then we loop throught the 100 neurons for each word
    for n in range(0, neuronCount):

        #add something random
        neuron.append(random.random())

    #add to our weights
    outputWeights.append(neuron)
    

#make one for loop training on everything!
for all in range(0, len(Inputs)):


    hiddenState = [] #our memory
    for i in range(0, neuronCount):
        hiddenState.append(0) #we init it with 100 0's



    #OK so all the code above init our nuerons and memeory for tranining
    #ok, now we need to match the ID and the words
    #we are gonna analize the mem, and our current word mem
    #this creates a mathamatical summery of this!
    currentWordMem = []

    for wordId in Inputs[all]:
        currentWordMem = weights[wordId]

        #ok, now we need to add our current mem to our hidden state

        for i in range(0, len(hiddenState)):
            hiddenState[i] = hiddenState[i] + currentWordMem[i]
        
    
    greatestScore = float('-inf')
    greatestScoreID = 0

    #ok, now we want to translate our thing back into a word, for our guess
    for i in range(0, wordCout):
        #grab the 100 output weights for the current word
        currentWordOutput = outputWeights[i]
        wordScore = 0

        #do the math
        for out in range(0, neuronCount):
            wordScore = wordScore + (hiddenState[out] * currentWordOutput[out])

        #check if the output is our highest score!
        if wordScore > greatestScore:
            greatestScore = wordScore
            greatestScoreID = i


        #grab the id for the correct awns

    targetId = Targets[all]
    #ok, now we check our wanted word, vs the word we got, and if it doesnt match, we adjust it and try again
    for out in range(0, neuronCount):
        outputWeights[targetId][out] = outputWeights[targetId][out] + learningRate

    #however we need to pen the bad guess
    if greatestScoreID != targetId:

        for out in range(0, neuronCount):
            outputWeights[greatestScoreID][out] = outputWeights[greatestScoreID][out] - learningRate

    if all % 1000 == 0 or all == len(Inputs) - 1:
        print(f"Progress: {all}/{len(Inputs)}")



#return the response
for step in range(0, 10):
    greatestScore = float('-inf')
    greatestScoreID = 0


    for i in range(0, wordCout):
        currentWordOutput = outputWeights[i]
        wordScore = 0

        for out in range(0, neuronCount):
            wordScore = wordScore + (hiddenState[out] * currentWordOutput[out])

        if wordScore > greatestScore:
            greatestScore = wordScore
            greatestScoreID = i

    
    print(Tokenizer.vocab[greatestScoreID])


#ok we are done, lets save the model

print("Saving!")

#save everything
model_data = {
    "weights": weights,
    "outputWeights": outputWeights,
    "vocab": Tokenizer.vocab,
    "wordID": Tokenizer.wordID
}

with open("poem_model.json", "w", encoding="utf-8") as f:
    json.dump(model_data, f)

"Done!"