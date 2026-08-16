import os, torch, random #imports, import the os, torch, and random

torch.set_num_threads(2)
torch.set_num_interop_threads(1) #set the threads up to prevent using a lot, as NEST is small


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "poem_model.txt")
WEIGHTS = os.path.join(BASE, "poem_weights.bin")

MAX_TOKENS, TOP_K = 120, 30 #how many tokens to generate, and how many tokens to randomly consider for the next word
REPEAT_PENALTY, RHYME_BOOST = 0.75, 0.25 #how much to penalize repeated words, and how much to boost rhyming words

print("loading...") #DEBUG

#LOAD THE MODEL -------------------------------->

with open(MODEL, encoding="utf-8", errors="ignore") as f: #open the model folder first
    word_count = int(f.readline()) #map the word count, as its the first line
    neurons = int(f.readline()) #map the neurons, as its the second line
    context = int(f.readline()) #map the context, as its the third line
    vocab_size = int(f.readline()) #map the vocab size, as its the fourth line

    f.readline()  # now we read the rhymes, which are stored as a list of word ids that rhyme with each other

    rhymes = {} # create a dir to hold the vocab of rhymes

    while True: #loop through the vocab, till we hit the END tag
        line = f.readline().strip() #read the line without whitespace
        if line == "END_RHYME": #if it is the end of the rhymes, we break
            break

        p = line.split() #converts the line ('example 1 2 3') into ["example", 1, 2, 3]
        if len(p) >= 3: #check to make sure there is enough info (not just one word, and not just two words)
            ids = list(map(int, p[1:])) #then we convert the list of strings into a list of integers, which are the word ids that rhyme with each other
            for x in ids: #takes the ids, and for each ids
                rhymes[x] = tuple(ids) # we group them together, giving the words a id rhyme group

    id2word, word2id = {}, {} #we create two dicts, one for id to word (123 -> "hello"), and one for word to id ("hello" -> 123)
    for _ in range(vocab_size): #reads through the whole vocab, and maps the words to their ids
        p = f.readline().strip().split(" ", 1) #splits it into the id, and the word, the one makes sure we only do it once. [as some have spaces in them]
        if len(p) != 2: #if the length of the list is not 2 (the ["hello", 123]), we skip it, as its not valid
            continue
        try: #attempt to convert the first part into an int, and the second part is the word
            wid, word = int(p[0]), p[1]
        except ValueError: #if it fails
            try: #attempt to convert the second part into an int, and the first part is the word
                word, wid = p[0], int(p[1])
            except ValueError: #it if it fails again, we skip it overall
                continue

        id2word[wid] = word #add it it the id to word dict
        word2id[word] = wid #add it to the word to id dict, word

input_n = context * word_count * neurons #calculate the number of input neurons, which is the context * word count * neurons
output_n = word_count * neurons #calculate the number of output neurons, which is the word count * neurons
total_n = input_n + output_n # calculate the total number of neurons, which is the input neurons + output neurons

#LOAD THE WEIGHTS -------------------------------->

print("loading weights...") #DEBUG

size = os.path.getsize(WEIGHTS) #Get the size of the weights file.

if size == total_n * 2: # if the size of the weights file is 2 times the total num of neurons
    dtype = torch.float16 # set the dtype to float16
elif size == total_n * 4: # if the size of the weights file is 4 times the total num of neurons
    dtype = torch.float32 # set the dtype to float32
elif size == total_n * 8: # if the size of the weights file is 8 times the total num of neurons
    dtype = torch.float64 # set the dtype to float64
else: #else -> if its not either of those
    raise RuntimeError(
        f"Bad weight size: {size:,} bytes" #DEBUG
    )

raw = torch.from_file( #open the file, as a pytorch tensor.
    WEIGHTS, #grab the weights
    dtype = dtype, # set the datatype to the one we just got
    size = total_n #set the size to the total number of neruons
)

weights = raw[:input_n].view(
    context, word_count, neurons
)

out_weights = raw[input_n:].view(
    word_count, neurons
).float()

# print( #DEBUG
#     f"Vocab: {word_count} | "
#     f"Neurons: {neurons} | "
#     f"Context: {context} | "
#     f"Model RAM-mapped: | "
#     f"{size / 1024**2:.1f} MB"
# )


bias = ( torch.arange( word_count, dtype=torch.float32) * -0.000001) #give each word a tiny bias, so the model does not always pick the same word.

common = list(range(1, min(500, word_count))) #handle the fallback, if the first word is not found, we pick a random word.


#DONE LOADING -------------------------------->

def generate_poem(prompt): #generate_poem

    prompt = prompt.strip().lower() #clean the prompt.
    if not prompt: #if the prompt is empty, we return nothing
        return ""

    rng = random.Random() #generate a random number, to insure we pick diffrent words each time

    first = word2id.get( prompt.split()[0], rng.choice(common)) #get the first word, and if we cannot find the prompt word, we pick a random word from the common word list

    if first <= 0 or first >= word_count: #if the first word is not in the vocab, pick a random one
        first = rng.choice(common)

    ids = [first] #hold the word ids, starting with the first one
    words = [id2word.get(first, "the")] #holds the actual word for printing
    used = {first: 1} #tracks how man times the word is used, adding the first word
    rhyme_target = None #we dont have a rhyme target yet.

    hidden = torch.zeros( neurons, dtype=torch.float32 ) #build the hidden layer, (the neurons)

    for _ in range(MAX_TOKENS - 1): # loop through each token, and build the next word - 1 for our first word
        hidden.zero_() #reset the hidden layer to 0 

        #grab the last context words, and add them to the hidden layer
        for pos, wid in enumerate( ids[-context:] ): #get the recent words
            if 0 < wid < word_count: #if the word id is valid, add it
                hidden += weights[pos, wid].float()

        hidden.clamp_(-1, 1) # clamp hidden between -1 and 1 to prevent any overflow

        scores = torch.mv( out_weights, hidden) #multiply the hidden layer by the output weight, to get a score for each word in the voacb

        scores += bias #add the bias to the scores
        scores[0] = -float("inf") #ban word 0, (thats an unknown word) so we dont want that

        # repetition
        for wid, count in used.items(): # for each word id, how many times its been used
            if 0 < wid < word_count: #check if its valid
                scores[wid] -= REPEAT_PENALTY * count #penalize the score for that word, on how many times its been used

        # rhyme
        if rhyme_target is not None: #if we have a rhyming target, the rhyme_target is the word id of the last word.
            for wid in rhymes.get(rhyme_target, () ):#pull words that rhyme with the target
                if 0 < wid < word_count: #if there valid words
                    scores[wid] += RHYME_BOOST #boost the score, to have it more likely to be picked

    
        scores += torch.randn_like(scores) * 0.001 #add a bit of randomness, to the scores, allowing a bit more randomness in generation

        top = torch.topk( scores, min(TOP_K, word_count - 1) ).indices.tolist() #then find the top candidates, and will chose from how many TOP_K words there are.

        if not top: #if there are no top things, we break
            break

        best = rng.choice(top[:5]) #select a random word from the top 5 candidates, from the word id's
        word = id2word.get(best, "?") #if the word id is NOT found, we make it a '?'

        if word == "?": #if it is the ? we want to break, as its unknown, and we dont want the model to keep creating as it will break
            break

        if word == "<NEWLINE>": #if the word is a new line
            if words[-1] == "<NEWLINE>": #if the last word was also a new line, we can jst skip
                continue

            rhyme_target = None #reset the rhyme target, before looking for the last word

            for wid in reversed(ids): # go through our generated word ID's backwards, because we want the most recent real word
                w = id2word.get(wid, "") # then once we get it, we convert the id back into a word
                if w != "<NEWLINE>": #insure its not a newline token
                    rhyme_target = wid #set the rhyme target to the last word id, so we can rhyme with it
                    break

            words.append(word) # append the generated words to it
            ids.append(best) #append the id's of the words
            continue #skip

        words.append(word) #if its not a newline, still append the words to it
        ids.append(best) #append the id's of the words
        used[best] = used.get(best, 0) + 1 # update the counter for repeated words

    lines, line = [], [] #now hold the compleated poem, and the lines being built

    for word in words: # go through every word
        if word == "<NEWLINE>": #if the word is a new line
            if line: #if the line is not empty
                lines.append(" ".join(line)) # we create a new line
            line = [] # set it back to blank
        else:
            line.append(word) #we just add the word onto the line

    if line: #if the line is not empty
        lines.append(" ".join(line)) #we add the last line to the lines

    return "\n".join(lines) #return the final join