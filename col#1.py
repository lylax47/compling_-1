import os
import sys    #for sys.exit() to help with debugging
import regex as re
import nltk
import string
import math

def clean(corp):
    n_corp = re.sub('<!-- The default annotation set -->(.|\n)*',r'', corp) # removes end half of xml tags
    n_corp = re.sub('<[^<]+>',r'', n_corp)  # removes remaining xml tags
    return n_corp

def calc_mean(dif_list):        # calculates mean of difference
    suum = sum(dif_list)
    mean = suum/len(dif_list)
    return mean

def std_dev(dif_list, mean):                   # calculates standard of deviation of differences
    square_list = [(x-mean)**2 for x in dif_list]
    suum = sum(square_list)
    dev = math.sqrt(suum/(len(square_list)-1))
    return dev


def values(texts, word, wind):
    co_tot_dic = {}
    indices_dif = {}
    stop = list(string.punctuation)
    stop.append('–')
    tokens = nltk.word_tokenize(texts)
    indices = ([i for i, j in enumerate(tokens) if j == word])  # finds indices of word in texts
    for i in indices:
        frame = tokens[(i-wind):(i+wind)]   # finds window of word based on given window value
        new_i = 0 + wind                    # index of main word within frame.
        for w in frame:
            ind = frame.index(w)
            dif = ind - new_i               # index dif
            if w in co_tot_dic.keys():      # finds and counts all possible collocates in every context
                co_tot_dic[w][0] += 1
                indices_dif[w].append(dif)      # will append all distance differences to a word
            else:
                co_tot_dic[w] = [1]
                indices_dif[w] = [dif]
    for k,v in list(co_tot_dic.items()):        # removes entries seen only three times and stop word/punctuation.
        if v[0] <= 3 or k in stop:    # maybe we can make this an option as well
            del co_tot_dic[k]
    for k in list(co_tot_dic.keys()):           # finds total counts for possible collocates
        co_tot_dic[k].append(len([c for c in tokens if c == str(k)]))          # append to dictionary list for word "k"
        k_mean = calc_mean(indices_dif[k])
        co_tot_dic[k].append(k_mean)                                           # append mean of coll k
        co_tot_dic[k].append(std_dev(indices_dif[k], k_mean))                  # append std of dev of coll k
    token_tot = len(tokens)
    word_tot = len(indices)
    print(co_tot_dic)
    #sys.exit()
    return (token_tot, word_tot, co_tot_dic)        # returns 3 value tuple with - token total, word total,\
    #  and dictionary --> {collocate:[co-occurrence count, total collocate count, mean dist, std_dev dist], ...}


def chi_sq():  
    w1 = float(vals[1])#key word in corpus without collocates 
    none = float(vals[0]) - w1 #all tokens in corpus without key word and collocate
    for token in list(vals[2].keys()):
        for val in list(vals[2].values()):
            w2 = float(val[1]-val[0]) #amount of every colocate in corpus
            both = float(val[0]) #amount of every collocation in corpus
        return(float(vals[0])*(both*none-w1*w2)**2/(both+w2)*(both+w1)*(both+none)*(none+w1))
# def loglike():
#                      #NB Please only return 100 best results!!!
#                      #NB and return results in dictionary format with collocates as keys and their values!!!
# def mutual_info():


corps = []
word = input("Please insert word:")             # input for word you want to search
wind = int(input("Please insert window size:")) # input for window size you want to search in
for root,dirs,files in os.walk("nanocorpus"):   # cleans all corpus texts in nanocorpus file
    for name in files:
        with open(os.path.join(root,name), 'r', encoding='utf-8') as corpus:
            corps.append(clean(corpus.read()))
texts = "\n".join(corps)                        # combines texts into one large text
vals = values(texts,word,wind)
chi_sq(vals)                                  #placeholders for finished stat functions
# loglike(vals)
# mutual(vals)
