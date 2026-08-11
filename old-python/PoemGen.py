#first we need to load the json
import json
#ok, open it up, using our poem_model.json

#we read with "r" dont do "w"
with open("poem_model.json", "r", encoding="utf-8") as f:
    model_data = json.load(f) #load that file



#load the weights, output weights, vocab, and word ID
neuronCount = 100
weights = model_data["weights"]
outputWeights = model_data["outputWeights"]
wordID = model_data["wordID"]

#ok, so our vocab has been turned into a strings, we need to convert them back
vocab = {}
for key, value in model_data["vocab"].items():
    vocab[int(key)] = value


wordCount = len(vocab) + 1 #we do +1, because we dont count 0 in the vocab list


