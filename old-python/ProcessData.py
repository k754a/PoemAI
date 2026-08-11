import pandas as pd

#because its a .csv file, first we are gonna convert our dataset to a txt file, removing some of the headers and stuff

#this funct will handle the data, for our main Poem Generator
def HandleData():
    #first lets open our file

    # p = pd.read_csv("poetry.csv", usecols=['content'])

    # print(p)

    ##each row is a poem, so lets conver it into a txt file, so we can work with it easier

    #for each poem in the content column, we add it to the file
    with open("poems.txt", "w", encoding="utf-8") as f:
        for poem in p['content']:
            #add the poem, and then make a new line
            f.write(poem + "\n")


    #a list of chars that we dont want
    remove_chars = r"""/\.,:;'+=-|"'!@#$%^&*())1234567890"""
    #this will also handle the removing unwanted punctuation and chars:
    #we use with open, as it handles closing the file for us
    finallines = ""
    with open("poems.txt", "r", encoding="utf-8") as f:
        #ok, for each line, in the file, lets remove the "/\.,:;'+=-| and whatever other chars there are
        for line in f:
            #the way it works, is it takes in a line, and attempts to translate an chars to whatever i set. we use this instead of replace
            #because it lets me do multiple making a transition table.
            cleanline = line.translate(str.maketrans("", "", remove_chars))
            finallines += cleanline;



        
    print(finallines)

    #the W mode, removes all text on open
    with open("poems.txt", "w") as file:
        file.write(finallines)


    print("Done!")









