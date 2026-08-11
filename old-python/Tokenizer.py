#ok this is our tokenizer.

#We are building a RNN from scratch, a type of ai network. im using a tutorial as ive never touched this before!
numTokens = [] #global
vocab = {} # save our vocab
wordID = {}

def Token():
    # we have our data, in a txt file, we need to split it and rank it from the most occurrences to the list.
    full = ""
    with open("poems.txt", "r", encoding="utf-8") as f:
        full = f.read()

    #now we have the full text, lets make a list

    words = full.split() #split into words

    word_counts = {} # Using a dictionary


    #for each word in words
    for word in words:

        if word in word_counts: #checks if we have this saved in our dict
            word_counts[word] +=1 #add one to the dict
        else: #add it to our dict
            word_counts[word] =1 #make a new dict, and set it to 1
        

    
    current_id = 1
    #now we want to assign an ID to each word, as we have the counts, but not the id, and the rank from biggest to smallest
    for word in word_counts.keys():
        vocab[current_id] = word # assign the vocab word!
        wordID[word] = current_id

        current_id +=1


    #ok, now we want to translate the poem

    for word in words:
        #we assign each word to a token that our id thing did
        numTokens.append(wordID[word])



    print("Done!")
    
