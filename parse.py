import os
import sys
import conllu
import pickle
import numpy as np

#GLOBAL DICTS
STARTPROB = {}
TRANSPROB = {}
EMISSIONPROB = {}
LISTOFTAGS = []

''' Calculate list of tags '''
def calcTagList(dataset):
    tagDict = {} #dictonary record number of upostag seen
    '''if upostag exists increment else create new and set to 1'''
    for each in dataset:
        for i in range(len(each) - 1):
            tag = each[i]['upostag'] #get upostag for word
            if tag not in tagDict:
                tagDict[tag] = 1
    '''Convert dict in to list of tags'''
    for each in tagDict:
        LISTOFTAGS.append(each)

''' Calculate starting probabilities '''
def calcStartProb(dataset):
    startDict = {} #dictonary record number of upostag seen
    startTotal = 0 #count how many upostag seen from dataset
    '''if upostag exists increment else create new and set to 1'''
    for each in dataset:
        tag = each[0]['upostag'] #get upostag for word
        startTotal += 1 #increment total counter
        if tag in startDict:
            startDict[tag] = startDict.get(tag) + 1
        else:
            startDict[tag] = 1
    '''Calculate start probability and store in global dict'''
    for each in startDict:
        prob = (startDict[each]/startTotal)
        STARTPROB[each] = prob

''' Calculate transitional probaility '''
def calcTransProb(dataset):
    transDict = {} #dictonary record number of transations from x->y
    transTotal = 0 #count how many transitions there are
    '''itr through sentence and record each instance of POS x->y '''
    for each in dataset:
        for i in range(len(each) - 1):
            transTotal += 1 #increment total counter
            tagX = each[i]['upostag']
            tagY = each[i+1]['upostag']
            transition = tagX + "_" + tagY #append x->y POS transition used as dict key
            '''if transition exists increment else create new and set to 1'''
            if transition in transDict:
                transDict[transition] = transDict.get(transition) + 1
            else:
                transDict[transition] = 1
    '''Calculate transition probability and store in global dict'''
    for each in transDict:
        prob = (transDict[each]/transTotal)
        TRANSPROB[each] = prob

''' Calculate emission probaility '''
def calcEmissionProb(dataset):
    emissionDict = {}
    for each in dataset:
        for i in range(len(each)):
            tag = each[i]['upostag']
            word = each[i]['form']
            ''' Logic handling nested dict '''
            #if tag not in dict then no word has been counted for that tag
            if tag not in emissionDict: 
                emissionDict[tag] = {}
                if word not in emissionDict[tag]:
                    emissionDict[tag][word] = 1
            #if tag is in dict check if word is counted; if not create new, else increment count
            if tag in emissionDict: 
                if word not in emissionDict[tag]:
                    emissionDict[tag][word] = 1
                else:
                    emissionDict[tag][word] = emissionDict[tag][word] + 1
    ''' calculate emission probability '''
    for upostag in emissionDict:
        tempTotal = 0
        EMISSIONPROB[upostag] = {}
        #calculate total words for current upostag
        for each in emissionDict[upostag]:
            tempTotal += emissionDict[upostag][each]
        #calculate probability
        for each in emissionDict[upostag]:
            EMISSIONPROB[upostag][each] = emissionDict[upostag][each] / tempTotal

def safeLog(x):
    # The probability of any possible event is a negative number, 0 is used to represent
    # impossible
    if x == 0:
        return 0
    else:
        return np.log(x)

def safeMultiply(x, y):
    if x == 0:
        return 0
    elif y == 0:
        return 0
    else:
        return x + y

''' Implementation of Hidden Markov model '''
def viterbiAlgo(sentence):
    # Uncomment this line to look at the sentence, it is "['What', 'if', 'Google', 'Morphed', 'Into', 'GoogleOS', '?']"
    print(sentence)
    # Uncomment this line to look at the emission probabilities - note that Morphed and GoogleOS is not in the dictionary at all
    # print(EMISSIONPROB)
    ''' init tables and array used in dynamic programming '''
    sentenceLen = len(sentence) - 1
    states = np.arange(len(LISTOFTAGS))
    table = np.zeros( (len(LISTOFTAGS), sentenceLen) )
    bktrac = np.zeros( (len(LISTOFTAGS), sentenceLen) , dtype = 'S5' )
    pred = np.zeros( (sentenceLen, ) )
    ''' Starting probability '''
    '''
        Since our table is an indexable list, and our probability is a un-indexable dict 
        we need a external counter variable(eachCounter)
    '''
    eachCounter = 0
    print(sentence[0])
    for each in LISTOFTAGS:
        try:
            table[eachCounter,0] = safeMultiply(safeLog(STARTPROB[each]), safeLog(EMISSIONPROB[each][sentence[0]]))
        except:
            table[eachCounter,0] = 0
        # Uncomment this to look at the initial probabilities (in log scale)
        print(each + "\t" + str(table[eachCounter, 0]))
        bktrac[eachCounter,0] = each
        eachCounter += 1
    # Uncomment this to group the probability in time groups
    print("----")

    ''' Dynamic programming '''
    for i in range(1, sentenceLen): #for each word fill in probability
        print(sentence[i])
        state1Row = 0
        for state1 in LISTOFTAGS:
            bestProb = 0
            bestPrev = "NULL"
            state2Row = 0
            for state2 in LISTOFTAGS:
                tranState = state2 + "_" + state1
                try:
                    logProb = safeMultiply(safeMultiply(table[state2Row, i-1], safeLog(TRANSPROB[tranState])), safeLog(EMISSIONPROB[state1][sentence[i]]))
                except Exception as e: 
                    logProb = 0
                if not logProb == 0:
                    if (bestProb == 0) or (logProb > bestProb):
                        bestProb = logProb
                        bestPrev = state2
                state2Row += 1
            table[state1Row, i] = bestProb
            bktrac[state1Row, i] = bestPrev
            # Uncomment this to look at the probabilities at time i (in log scale)
            print(state1 + "\t" + str(table[state1Row, i]))
            state1Row += 1
        # Uncomment this to group the probability in time groups
        print("----")

if __name__ == "__main__":
    ''' handle command line arguments '''
    if len(sys.argv) != 3:
        print("Usage: python hmm.py --<train/test> <file>")
        sys.exit()
    function = sys.argv[1] #train or test data
    path = sys.argv[2] #path to data file
    #open conllu file for reading
    data_file = open(path, "r", encoding="utf-8")

    ''' we can either train from data, or test our result from data '''
    if function == "--train":
        dataset = []
        #store each tokenlist in dataset list
        for tokenlist in conllu.parse_incr(data_file):
            dataset.append(tokenlist)
        ''' Calculate relevent probailities '''
        calcTagList(dataset)
        calcStartProb(dataset)
        calcTransProb(dataset)
        calcEmissionProb(dataset)
        #print dictionary for debugging
        print(LISTOFTAGS)
        print(STARTPROB)
        print(TRANSPROB)
        print(EMISSIONPROB)

        #store DICT in pickle file
        with open('listoftag.pickle', 'wb') as tagOut:
            pickle.dump(LISTOFTAGS, tagOut)
        with open('startprob.pickle', 'wb') as startOut:
            pickle.dump(STARTPROB, startOut)
        with open('transprob.pickle', 'wb') as transOut:
            pickle.dump(TRANSPROB, transOut)
        with open('emissionprob.pickle', 'wb') as emissionOut:
            pickle.dump(EMISSIONPROB, emissionOut)

    ''' we can either train from data, or test our result from data '''
    if function == "--test":
        ''' load in pickled probability dictonary '''
        #list of possible tags
        tagIn = open('listoftag.pickle','rb')
        LISTOFTAGS = pickle.load(tagIn)
        #emission probability
        emissionIn = open('emissionprob.pickle','rb')
        EMISSIONPROB = pickle.load(emissionIn)
        #starting probability
        startIn = open('startprob.pickle','rb')
        STARTPROB = pickle.load(startIn)
        #transisition probability
        transIn = open('transprob.pickle','rb')
        TRANSPROB = pickle.load(transIn)

        for tokenlist in conllu.parse_incr(data_file):
            wordList = []
            tagList = []
            for word in tokenlist:
                wordList.append(word['form'])
                tagList.append(word['upostag'])
            viterbiAlgo(wordList)
            break