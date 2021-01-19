import nltk
from nltk.stem import PorterStemmer
import sys
import xml.etree.ElementTree as ET
import re
import math
from collections import OrderedDict
import operator
import numpy as np
from numpy import linalg as LA
stop_list = None

# function to check only 'letters', ''' and '-'
def checkForCharacters(word):
    if(word.isalpha()):
        return True
    for letter in word:
        if letter in "~`!@#$%^&*()_=+|\\]{[}:;/\"<>,.?1234567890":
            return False
    return True


def main():
    
    #Managing the arguments
    if str(sys.argv[1]=="--score"):
        if str(sys.argv[2]) == "BM25":
            print("BM25")
            g=1
        elif str(sys.argv[2]) == "OkapiTF":
            print("OkapiTF")
            g=3
        elif str(sys.argv[2]) == "TF-IDF":
            print("TF-IDF")
            g=4
        elif str(sys.argv[2]) == "Jelinek-Mercer":
            print("Jelinek-Mercer")
            g=2
        else:
            print("Invalid Evaluation Method \nsetting to BM25 \nAvailable options are BM25, OkapiTF, TF-IDF and Jelinek-Mercer\n")
            g=1
    else:
        print("Invalid Method \nsetting to BM25 \nAvailable options are BM25, OkapiTF, TF-IDF and Jelinek-Mercer\n")
        g=1

    filename = str(sys.argv[3])

    # filename = "ranking.txt"
    # g=1
    # reading stop-words
    global stop_list
    f2 = open("stoplist.txt", "r")
    if f2.mode == 'r':
        stop_list = f2.read()
        stop_list = nltk.word_tokenize(stop_list)
        f2.close()

    # reading index and term-ids
    f = open("termids.txt", "r", encoding="iso-8859-1")
    f2 = open("term_index.txt", "r")
    if f2.mode == 'r' and f.mode == 'r':
        termIds = f.read()
        termIds = nltk.word_tokenize(termIds)
        lines = f2.readlines()
        f.close()
        f2.close()

    # reading doc-lengths and doc-ids
    f = open("docids.txt", "r")
    f2 = open("docLengths.txt", "r")
    if f2.mode == 'r' and f.mode == 'r':
        documentIds = f.read()
        documentIds = nltk.word_tokenize(documentIds)
        lengths = f2.readlines()
        f.close()
        f2.close()

    # opening ranking.txt to write ranks
    ranking = open(filename, "w+")
    ranking.close()

    # finding average length of docs
    avgLen = 0
    totaldocs = len(lengths)
    for l in range(totaldocs):
        avgLen += int(lengths[l])
    avgLen = avgLen/totaldocs

    # finding total word count in corpus
    wordcount = 0
    for t in lines:
        freq = nltk.word_tokenize(str(t))
        wordcount += int(freq[1])

    # parsing query doc
    tree = ET.parse("topics.xml")
    root = tree.getroot()
    
    
    # loop for getting the average length of the queries
    gt = 0
    queryLengthCount = 0
    query_text_array =  []
    for child in root:
        x = root.find("./topic[@number=\""+child.attrib['number']+"\"]/")
        wordList = x.text.lower().rstrip().split(' ')
        text = x.text
    
        # tokenizing text
        text = nltk.word_tokenize(text)
    
        ps = PorterStemmer()
        final_text = []
    
        # adding words to final text if it conatins only 'letters', ''' and '-'
        for word in text:
            if (checkForCharacters(word) and re.match("[A-Za-z]", word)):
                if word not in stop_list:
                    final_text.append(ps.stem(word))
        #print(len(final_text))
        queryLengthCount += len(final_text)
        gt += 1
        query_text_array.append(final_text)
        
    avgQueryLength = queryLengthCount/gt
    
    # loop for each query
    for child in root:
        x = root.find("./topic[@number=\""+child.attrib['number']+"\"]/")
        wordList = x.text.lower().rstrip().split(' ')
        print(child.attrib['number'], x.text)
        text = x.text

        # tokenizing text
        text = nltk.word_tokenize(text)

        ps = PorterStemmer()
        final_text = []

        # adding words to final text if it conatins only 'letters', ''' and '-'
        for word in text:
            if (checkForCharacters(word) and re.match("[A-Za-z]", word)):
                if word not in stop_list:
                    final_text.append(ps.stem(word))

        score = {}
        query = list(range(len(final_text)))
        queryLength = len(final_text)
        count = 0
        q_score = []
        d_score = {}
        # loop for each query term
        for term in final_text:

            # in-case the word is found
            if term in termIds:

                # reading the index of the term from file
                term_index = termIds.index(term)

                # reading the term_id
                term_id = int(termIds[term_index - 1])

                # tokenizing the exact line where the term index is placed
                line = lines[term_id]
                docIds = (str(line))
                docIds = [x.split(',') for x in docIds.split(' ')]

                # calculating frequecy of term in all documents
                query[count] = {}
                num = 0
                for a in range(len(docIds)-3):
                    num += int(docIds[a+3][0])
                    if num in query[count]:
                        query[count][num] += 1
                    else:
                        query[count][num] = 1

            # calculating score for each doc
            docfreq = len(query[count])
            for d in query[count]:
                #g = 3
                # Okapi BM25 scoring  str(sys.argv[1]) == "BM25"
                
                if (g==1):
                    K = 1.2*((1-0.75)+0.75*(int(lengths[d])/avgLen))
                    result = math.log(((totaldocs + 0.5)/docfreq+0.5),2)
                    result *= ((1+1.2)*query[count][d])/(K+query[count][d])
                    result *= ((1+900)*1)/900+1
                    if d in score:
                        score[d] += result
                    else:
                        score[d] = result

                # Jelinik Mercer
                elif(g==2):
                    N = int(lengths[d])
                    myLambda =  0.6
                    freqDoc = query[count][d]
                    freqCorpus = int(docIds[1][0])
                    result = myLambda*(freqDoc/N)
                    result2 = (1-myLambda)*(freqCorpus/wordcount)
                    result3 = result+result2
                    if d in score:
                        score[d] *= result3
                    else:
                        score[d] = result3
                # Okapi TF
                elif(g==3):
                    termFreq = query[count][d]
                    okt_f = termFreq/(termFreq+0.5+(1.5*(int(lengths[d])/avgLen)))
                    if d in d_score:
                        d_score[d][count]= okt_f
                    else:
                        buckets = []
                        for i in range(0,queryLength):
                            buckets.append(0)
                        d_score[d] = buckets
                        d_score[d][count]= okt_f
                #TF-IDF
                elif(g==4):
                    termFreq = query[count][d]
                    okt_f = termFreq/(termFreq+0.5+(1.5*(int(lengths[d])/avgLen)))
                    if d in d_score:
                        d_score[d][count]= okt_f
                    else:
                        buckets = []
                        for i in range(0,queryLength):
                            buckets.append(0)
                        d_score[d] = buckets
                        result = okt_f*math.log(totaldocs/docfreq)
                        d_score[d][count]= okt_f

            count += 1
        if g==3:
            oktf_q = 1/(1+0.5+(1.5*(queryLength/avgQueryLength)))
            for i in range(0,queryLength):
                q_score.append(oktf_q)
            for d_s in d_score:
                key1 = d_s
                oo = d_score[d_s]
                p_d = LA.norm(oo)
                p_q = LA.norm(q_score)
                dtprdct = np.dot(oo,q_score)
                result = dtprdct/(p_d*p_q)
                score[d_s]=result
            #print("ok")
        elif g==4:
            oktf_q = 1/(1+0.5+(1.5*(queryLength/avgQueryLength)))
            for query_word_n in final_text:

                # in-case the word is found
                if query_word_n in termIds: 
                    h = 0
                    for q_lines in query_text_array:
                        if query_word_n in q_lines:
                            h+=1
                    
                    q_score.append(oktf_q*math.log(gt/h))
           # print("ok")
                
            for d_s in d_score:
                key1 = d_s
                oo = d_score[d_s]
                p_d = LA.norm(oo)
                p_q = LA.norm(q_score)
                dtprdct = np.dot(oo,q_score)
                result = dtprdct/(p_d*p_q)
                score[d_s]=result
            print("ok")
        # sorting on the basis of score
        scoring = sorted(
            score.items(), key=operator.itemgetter(1), reverse=True)
        counter = 1
        ranking = open(filename, "a+")

        # printing and writing in ranking.txt
        for s in scoring:
            fileIdentifier_index = documentIds.index(str(s[0])) 
            file_identifier = str(documentIds[fileIdentifier_index + 1])
            print(child.attrib['number'] + " 0 " + file_identifier + " " +
                  str(counter) + " " + str(s[1]) + " run1")
            ranking.write(child.attrib['number'] + " 0 " + file_identifier + " " +
                          str(counter) + " " + str(s[1]) + " run1\n")
            counter += 1
        ranking.close()


if __name__ == "__main__":
    main()
