import os
import sys
import conllu
import pickle
import numpy as np

#GLOBAL DICTS
STARTPROB = {} #probability of starting with this pos
TRANSPROB = {} #probability of transitioning from pos X -> Y
EMISSIONPROB = {} #probability of word coming emitting from particular pos
TAGPROB = {} #probabililty of seeing this tag

#PENALTY FOR UNKNOWN WORDS
UNKOWN_PENALTY = 0.9 #unknow words suffer a 10% penality to probaility level

''' Calculate list of tags '''
def calcTagProb(dataset):
    tagDict = {} #dictonary record number of upostag seen
    tagTotal = 0
    '''if upostag exists increment else create new and set to 1'''
    for each in dataset:
        for i in range(len(each) - 1):
            tag = each[i]['upostag'] #get upostag for word
            if tag in tagDict:
                tagDict[tag] = tagDict.get(tag) + 1
            else:
                tagDict[tag] = 1
            tagTotal += 1
    '''Convert dict in to list of tags'''
    for each in tagDict:
        prob = (tagDict[each]/tagTotal)
        TAGPROB[each] = prob

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
            word = each[i]['form'].lower()
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

def simpleStemming(word):
    ''' Very naive stemming technique, remove 1 char at a time from back ,do samething from front
        return a list of each char removed and brute force to see if variation exits.
        sort list so that longest words appear in the front, so we dont automaticlly assume the word is
        a DET like A.
    '''
    morphList = []
    for i in range(len(word)):
        morphList.append(word[:i*-1])
        morphList.append(word[i:])
    morphList = list(dict.fromkeys(morphList)) #removes duplicates from list
    morphList.sort(key=len, reverse=True) #sort by word length
    morphList = morphList[1:-1] #remove front and back
    morphList = [i for i in morphList if len(i) >= 2]
    return morphList

''' Implementation of Hidden Markov model '''
def viterbiAlgo(sentence, tags):
    #print(sentence)#DEBUG
    ''' init tables and array used in dynamic programming '''
    sentenceLen = len(sentence)
    states = np.arange(len(TAGPROB))
    table = np.zeros( (len(TAGPROB), sentenceLen) )
    pred = []
    ''' Starting probability '''
    '''
        Since our table is an indexable list, and our probability is a un-indexable dict
        we need a external counter variable(eachCounter)
    '''
    eachCounter = 0
    #print("[ " +sentence[0] + " ] " + tags[0])#DEBUG
    for each in TAGPROB:
        try:
            table[eachCounter,0] = safeMultiply(safeLog(STARTPROB[each]), safeLog(EMISSIONPROB[each][sentence[0]]))
        except:
            morphList = simpleStemming(sentence[0])
            for wordMorph in morphList:
                try:
                    table[eachCounter,0] = safeMultiply(safeLog(STARTPROB[each]), safeLog(EMISSIONPROB[each][wordMorph] * UNKOWN_PENALTY * TAGPROB[each]))
                    break
                except:
                    continue
        #print(each + "\t" + str(table[eachCounter, 0]))#DEBUG
        eachCounter += 1
    #print("----") #DEBUG

    ''' Dynamic programming '''
    for i in range(1, sentenceLen): #for each word fill in probability
        #print("[ " +sentence[i] + " ] " + tags[i])#DEBUG
        state1Row = 0
        for state1 in TAGPROB:
            bestProb = 0
            state2Row = 0
            for state2 in TAGPROB:
                tranState = state2 + "_" + state1
                logProb = 0
                try:
                    logProb = safeMultiply(safeMultiply(table[state2Row, i-1], safeLog(TRANSPROB[tranState])), safeLog(EMISSIONPROB[state1][sentence[i]]))
                except Exception as e:
                    morphList = simpleStemming(sentence[i])
                    for wordMorph in morphList:
                        try:
                            logProb = safeMultiply(safeMultiply(table[state2Row, i-1], safeLog(TRANSPROB[tranState])), safeLog(EMISSIONPROB[state1][wordMorph] * UNKOWN_PENALTY * TAGPROB[state1]))
                            break
                        except:
                            continue
                if not logProb == 0:
                    if (bestProb == 0) or (logProb > bestProb):
                        bestProb = logProb
                state2Row += 1
            table[state1Row, i] = bestProb
            #print(state1 + "\t" + str(table[state1Row, i]))#DEBUG
            state1Row += 1
        #print("----")#DEBUG

    for i in range(sentenceLen):
        possibleTags = table[:,i]
        maxProb = np.max(possibleTags[np.nonzero(possibleTags)])
        idx = possibleTags.tolist().index(maxProb)
        tagList = list(TAGPROB.keys())
        pred.append(tagList[idx])

    return pred

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
        calcTagProb(dataset)
        calcStartProb(dataset)
        calcTransProb(dataset)
        calcEmissionProb(dataset)

        #store DICT in pickle file
        with open('tagprob.pickle', 'wb') as tagOut:
            pickle.dump(TAGPROB, tagOut)
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
        tagIn = open('tagprob.pickle','rb')
        TAGPROB = pickle.load(tagIn)
        #emission probability
        emissionIn = open('emissionprob.pickle','rb')
        EMISSIONPROB = pickle.load(emissionIn)
        #starting probability
        startIn = open('startprob.pickle','rb')
        STARTPROB = pickle.load(startIn)
        #transisition probability
        transIn = open('transprob.pickle','rb')
        TRANSPROB = pickle.load(transIn)

        counter = 0
        for tokenlist in conllu.parse_incr(data_file):
            if counter == 2:
                break
            wordList = []
            tagList = []
            for word in tokenlist:
                wordList.append(word['form'].lower())
                tagList.append(word['upostag'])
            pred = viterbiAlgo(wordList, tagList)

            counter += 1
