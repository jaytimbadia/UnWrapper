
# coding: utf-8

# In[1]:


c=open('C:\\Users\jay.timbadia\Downloads\Multiutility_lrclassifier.txt', encoding='utf-8', errors='ignore')
import pandas as pd
import re
p = c.readlines()
y = p[0].split('LRRetain ')
print(y[1])
dataf = pd.DataFrame()
for i in p:
    t = re.findall(r'(LRRetain|LRReview)', i)
    if t[0] == 'LRReview':
        o = i.split('LRReview ')
        sentence = o[1]
        label = 'LRReview'
        dataf = dataf.append({'Sentence': sentence, 'Labels': label}, ignore_index=True)
        
    elif t[0] == 'LRRetain':
        o = i.split('LRRetain ')
        sentence = o[1]
        label = 'LRRetain'
        dataf = dataf.append({'Sentence': sentence, 'Labels': label}, ignore_index=True)
        
dataf['Labels'] = dataf['Labels'].map({'LRReview':1, 'LRRetain':0})
dataf.head()


# In[2]:


import nltk
from nltk.corpus import conll2000

class ConsecutiveNPChunkTagger(nltk.TaggerI): 
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm='iis', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


# In[3]:


train_sents = conll2000.chunked_sents('train.txt')
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos, 
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)} 

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))
chunker = ConsecutiveNPChunker(train_sents[:100])


# In[4]:


from nltk.tokenize import word_tokenize
text = word_tokenize(dataf['Sentence'][0])
tagged_sents = nltk.pos_tag(text)
c = chunker.parse(tagged_sents)


# In[5]:


sent_0 = dataf.loc[dataf['Labels'] == 0]['Sentence'][:50].tolist()
sent_1 = dataf.loc[dataf['Labels'] == 1]['Sentence'][:50].tolist()
import string
punclist = list(string.punctuation) + ["'", "'s", "-"]
string.punctuation


# In[6]:


def Func(sentences):
    sent_tree = []
    tagged_s = []
    for i in sentences:
        text1 = []
        text = word_tokenize(i)
        for i in text:
            if i not in punclist:
                text1.append(i)
            else: continue
        tagged_sent = nltk.pos_tag(text1)
        tagged_s.append(tagged_sent)
        c = chunker.parse(tagged_sent)
        sent_tree.append(c)
    return tagged_s, sent_tree

tagged_sentence0, sent_0tree = Func(sent_0)
tagged_sentence1, sent_1tree = Func(sent_1)


# c= []
# tokens, pos = zip(*sent_0tree[6].leaves())
# fre = []
# for node in sent_0tree[6]:
#     if hasattr(node, 'label'):
#         fre.append({node.label():list(map(lambda t:t[1], node.leaves()))})
#     else:
#         fre.append({'XX':[node[1]]})
# fre

import cv2
def img_to_sig(arr):

    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig

arr1 = np.array([[1, 2, 3]])

arr2 = np.array([[3, 2, 1]])

sig1 = img_to_sig(arr1)
sig2 = img_to_sig(arr2)

print(sig1)
print(sig2)

dist, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)

print(dist)
print(_)
print(flow)


# In[24]:


import itertools
ROOT = 'ROOT'
tree = c
from collections import defaultdict
def getNodes(parent):
    frequency = []
    for node in parent:
        if hasattr(node, 'label'):
            frequency.append({node.label():list(map(lambda t:t[1], node.leaves()))})
        else:
            frequency.append({'XX':[node[1]]})
    return frequency

dictionary0 = []
dictionary1 = []

for i in sent_0tree:
    a = getNodes(i)
    dictionary0.append(a)

for i in sent_1tree:
    a = getNodes(i)
    dictionary1.append(a)

dictr0 = list(itertools.chain(*dictionary0))
dictr1 = list(itertools.chain(*dictionary1))
dictionary0


# In[25]:


import pandas as pd
import numpy as np

df0 = pd.DataFrame(dictr0)
df1 = pd.DataFrame(dictr1)
df0 = df0.replace(np.nan, '', regex=True)
df1 = df1.replace(np.nan, '', regex=True)

def createFilteredList(df, string):
    npl = df[string].values.tolist()
    npl = list(filter(None, npl))
    npl = list(filter(lambda x: len(x) > 1, npl))
    return npl

npl0 = createFilteredList(df0, 'NP')
vpl0 = createFilteredList(df0, 'VP')
ppl0 = createFilteredList(df0, 'PP')

npl1 = createFilteredList(df1, 'NP')
vpl1 = createFilteredList(df1, 'VP')
ppl1 = createFilteredList(df1, 'PP')

npl0


# In[26]:


def getFreqCount(seq, string):
    list1 , list2 = [], []
    for i in seq:
        if i not in list1:
            list1.append(i)
        else: continue

    for j in list1:
        count = 0
        for k in seq:
            if k == j:
                count+=1
        list2.append([count,j, string])
    return list2


# In[27]:


nlpcount_0 = getFreqCount(npl0, 'NP')
vplcount_0 = getFreqCount(vpl0, 'VP')
pplcount_0 = getFreqCount(ppl0, 'pp')

nlpcount_1 = getFreqCount(npl1, 'NP')
vplcount_1 = getFreqCount(vpl1, 'VP')
pplcount_1 = getFreqCount(ppl1, 'NP')
nlpcount_0


# In[28]:


def Convert2Frame(j ,k ,l):
    if j:
        x = pd.DataFrame(j).rename(columns = {0:'Count', 1:'Phrases', 2: 'Type'}).sort_values(by=['Count'], ascending=False)
        x = x[x['Count'] > 5]
    if k:
        y = pd.DataFrame(k).rename(columns = {0:'Count', 1:'Phrases', 2: 'Type'}).sort_values(by=['Count'], ascending=False)
        y = y[y['Count'] > 5]
    if l:
        z = pd.DataFrame(l).rename(columns = {0:'Count', 1:'Phrases', 2: 'Type'}).sort_values(by=['Count'], ascending=False)
        z = z[z['Count'] > 5]
    return pd.concat([x, y])

total1 = Convert2Frame(nlpcount_1, vplcount_1, pplcount_1)
total0 = Convert2Frame(nlpcount_0, vplcount_0, pplcount_0)


# In[14]:


total0


# In[15]:


total1


# In[29]:


def WindowPhraseFormation(df, dictionary):
    rightlist, leftlist, alllist = [], [], []
    for index, row in df.iterrows():
        p= {row['Type']:row['Phrases']}
        for sent in dictionary:
            for phrases in sent:
                if phrases == p:
                    idx = sent.index(phrases)
                    if idx == 0:
                        a = [p, sent[idx+1], sent[idx+2], sent[idx+3]]
                        a = list(map(lambda x:list(x.values()) ,a))
                        rightlist.append(a)
                    elif idx == len(sent) - 1:
                        b = [sent[idx-3], sent[idx-2], sent[idx-1], p]
                        b = list(map(lambda x:list(x.values()) ,b))
                        leftlist.append(b)
                    else:
                        if idx == len(sent) - 2:
                            c = [sent[idx-3], sent[idx-2], sent[idx-1], p, sent[idx+1]]
                        elif idx == len(sent) - 3:
                            c = [sent[idx-3], sent[idx-2], sent[idx-1], p, sent[idx+1], sent[idx+2]]
                        else:
                            c = [sent[idx-3], sent[idx-2], sent[idx-1], p, sent[idx+1], sent[idx+2], sent[idx+3]]
                        c = list(map(lambda x:list(x.values()) ,c))
                        alllist.append(c)
    return list(filter(None, rightlist)), list(filter(None, leftlist)), list(filter(None, alllist))   
                
rightlist0, leftlist0,  alllist0 = WindowPhraseFormation(total0, dictionary0)
rightlist1, leftlist1,  alllist1 = WindowPhraseFormation(total1, dictionary1)

def MakeCombined(seq, position):
    all_df0 = pd.DataFrame(seq)
    all_df0 = all_df0.replace(np.nan, '', regex=True)
    if position == 'Center':
        all_df0.rename(columns = {0:'Left3', 1:'Left2', 2:'Left1', 3:'Center', 4:'Right1', 5:'Right2', 6:'Right3'}, inplace = True)
        all_df0['combined'] = all_df0.apply(lambda x: list([x['Left3'], x['Left2'], x['Left1'], x['Center'], x['Right1'], x['Right2'], x['Right3']]),axis=1)
        all_df0.drop(['Left3', 'Left2', 'Left1', 'Center', 'Right1', 'Right2', 'Right3'], axis=1, inplace = True)
    elif position == 'Left':
        all_df0.rename(columns = {0:'Left3', 1:'Left2', 2:'Left1', 3:'Center'}, inplace = True)
        all_df0['combined'] = all_df0.apply(lambda x: list([x['Left3'], x['Left2'], x['Left1'], x['Center']]),axis=1)
        all_df0.drop(['Left3', 'Left2', 'Left1', 'Center'], axis=1, inplace = True)
    else:
        all_df0.rename(columns = {0:'Center', 1:'Right1', 2:'Right2', 3:'Right3'}, inplace = True)
        all_df0['combined'] = all_df0.apply(lambda x: list([x['Center'], x['Right1'], x['Right2'], x['Right3']]),axis=1)
        all_df0.drop(['Center', 'Right1', 'Right2', 'Right3'], axis=1, inplace = True)
                                                               
    all_df0['combined'] = all_df0['combined'].apply(lambda x: list(itertools.chain(*x)))
    return all_df0
  
center0 = MakeCombined(alllist0, 'Center')    
center1 = MakeCombined(alllist1, 'Center')
left0 = MakeCombined(leftlist0, 'Left')
left1 = MakeCombined(leftlist1, 'Left')
right0 = MakeCombined(rightlist0, 'Right')
right1 = MakeCombined(rightlist1, 'Right')

all_0 = pd.concat([center0, left0, right0], axis=0, ignore_index=True)
all_1 = pd.concat([center1, left1, right1], axis=0, ignore_index=True)
all_0


# In[30]:


# def fUNc(seq):
#     t= pd.DataFrame(seq)
#     t[3] = t[3].apply(lambda x:list(np.squeeze(x)))
#     all_middle = []
#     for index, row in t.iterrows():
#         if row[3] not in all_middle:
#             all_middle.append(row[1])
#         else:
#             continue
#     return all_middle

# all_middle0 = fUNc(alllist0)
# all_middle1 = fUNc(alllist1)
# all_middle0 = list(itertools.chain(*all_middle0))
# all_middle1 = list(itertools.chain(*all_middle1))

def Making(seq):
    str1 = ''
    a = len(seq)
    if a == 1:
        str1 += '<' + seq[0] + '>' + '|'
    else:
        for k in range(a):
            str1 += '<' + seq[k] + '>'
        str1 += '|'
    return str1[:-1]

def MakingChunks(df):
    list1 = []
    for index, row in df.iterrows():
        str2 = ''
        p = len(row['combined'])
        for i in range(p):
            y = Making(row['combined'][i])
            str2 += y
        list1.append(str2)
    return list1

list_0 = MakingChunks(all_0)
list_1 = MakingChunks(all_1)
list_0 = list(set(list_0))
list_1 = list(set(list_1))
list_0

list0 = list(map(lambda x:'CHUNK1 :' + '{' + x + '}' ,list_0))
list1 = list(map(lambda x:'CHUNK1 :' + '{' + x + '}' ,list_1))


# In[66]:


def GetSentence(tagged_doc, seq):
    chunks = []
    for j in seq:
        cp1 = nltk.RegexpParser(j)
        for i in tagged_doc:
            tree = cp1.parse(i)
            for subtree in tree.subtrees():
                if subtree.label() == 'CHUNK1':
                    chunks.append(subtree.leaves())
    return chunks  

chunks0 = GetSentence(tagged_sentence0, list0)
chunks1 = GetSentence(tagged_sentence1, list1)
chunks0
se0, se1= [], []
for i in chunks0:
    s0 = []
    for j in i:
        s0.append(j[0])
    se0.append(s0)
for i in chunks1:
    s1 = []
    for j in i:
        s1.append(j[0])
    se1.append(s1)
l1 = [1] * len(se1)
l0 = [0] * len(se0)
df = pd.DataFrame({'col':se1, 'label':l1})
df0 = pd.DataFrame({'col':se0, 'label':l0})
total_df = pd.concat([df, df0],axis=0, ignore_index=True, )
from sklearn.utils import shuffle
total_df = shuffle(total_df)
total_df

def f(row):
    return ' '.join(row)
total_df['Joined'] = total_df['col'].apply(f)
total_df.drop(['col'], axis =1, inplace=True)
total_df


# In[166]:


import nltk.corpus
from nltk.tokenize import WordPunctTokenizer
import nltk.stem.snowball
from nltk.corpus import wordnet
from nltk.chunk import ne_chunk
import string

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

tokenizer = WordPunctTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

stemmer = nltk.stem.snowball.SnowballStemmer('english')
def Match(tagged_sent0, tagged_sent1, oursent, threshold=0.5):
        
        words0, words1 = [], []
        for i in tagged_sent0:
            wrd = list(map(lambda x:x[0], i))
            words0.append(wrd)
            
        for i in tagged_sent1:
            wrd = list(map(lambda x:x[0], i))
            words1.append(wrd)
            
        oursent = tokenizer.tokenize(oursent)
            
        
        lemmae_a = [[lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in pos_a] for pos_a in words0]
        lemmae_b = [[lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in pos_b] for pos_b in words1]
        lemmae_our = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in oursent]
        
        ne_tree_a = list(map(lambda a: ne_chunk(nltk.pos_tag(a)), words0))
        ne_tree_b = list(map(lambda b: ne_chunk(nltk.pos_tag(b)), words1))
        ne_tree_our = ne_chunk(nltk.pos_tag(oursent))
        
        f, f1, our = [], [], []
        for node in ne_tree_our:
            if hasattr(node, 'label'):
                our.append({node.label():list(map(lambda t:t[0], node.leaves()))})
                    
        for node1 in ne_tree_a:
            for node in node1:
                if hasattr(node, 'label'):
                    f.append({node.label():list(map(lambda t:t[0], node.leaves()))})
                    
        
        for node1 in ne_tree_b:
            for node in node1:
                if hasattr(node, 'label'):
                    f1.append({node.label():list(map(lambda t:t[0], node.leaves()))})
                    
        for i in our:
            o = list(i.keys())
            for j in o:
                if j in ['ORGANIZATION', 'PERSON', 'GPE']:
                    u = len(list(i.values()))
                    if u > 1:
                        for k in range(u):
                            lemmae_our.remove(i[j][k].lower())
                    else:
                        lemmae_our.remove(i[j][0].lower())
        
        for i in f:
            o = list(i.keys())
            for h in lemmae_a:
                for j in o:
                    if j in ['ORGANIZATION', 'PERSON', 'GPE']:
                        u = len(list(i.values()))
                        if u > 1:
                            for k in range(u):
                                if i[j][k].lower() in h:
                                    h.remove(i[j][k].lower())
                        else:
                            if i[j][0].lower() in h:
                                h.remove(i[j][0].lower())
        
        for i in f1:
            o = list(i.keys())
            for h in lemmae_b:
                for j in o:
                    if j in ['ORGANIZATION', 'PERSON', 'GPE']:
                        u = len(list(i.values()))
                        if u > 1:
                            for k in range(u):
                                if i[j][k].lower() in h:
                                    h.remove(i[j][k].lower())
                        else:
                            if i[j][0].lower() in h:
                                h.remove(i[j][0].lower())

        stems_a = [[stemmer.stem(token) for token in i] for i in lemmae_a]
        stems_b = [[stemmer.stem(token) for token in i] for i in lemmae_b]
#         stems_our = [stemmer.stem(token) for token in lemmae_our] 
        stems_a = list(itertools.chain(*lemmae_a))
        stems_b = list(itertools.chain(*lemmae_b))
#         print(stems_our)
        # 1 - def - review
        nondef_ratio = len(set(stems_a).intersection(lemmae_our)) / float(len(set(stems_a).union(lemmae_our)))
        def_ratio = len(set(stems_b).intersection(lemmae_our)) / float(len(set(stems_b).union(lemmae_our)))
        return nondef_ratio, def_ratio
    
oursent = 'merger has been proposed by company Google & Microsoft'
# oursent = 'there will certainly a merger between company Google and Microsoft'
nondef, defi = Match(tagged_sentence0, tagged_sentence1, oursent)
print(-np.log(defi))
print(-np.log(nondef))


# In[65]:


from collections import Counter

def ListFormation(seq, seq2):
    all_l, all_r = [], []
    for pattern in seq:
        left_list, right_list = [], []
        t = pd.DataFrame(seq2)
        t[3] = t[3].apply(lambda x:list(np.squeeze(x)))
        for index, row in t.iterrows():
            if row[3] == pattern:
                left_list.append((row[2], row[1], row[0]))
                right_list.append((row[4], row[5], row[6]))
        all_l.append(left_list)
        all_r.append(right_list)

    def GetListAndFreq(seq1):
        all_, a = [], []
        for i in seq1:    
            d = list(itertools.chain(*i))
            all_.append(d)
#             m = list(map(lambda x:tuple(x) ,d))
#             m = Counter(m)
#             all_freq.append(m)
        return all_
    
    return all_l, all_r, GetListAndFreq

all_l0, all_r0, func0 = ListFormation(all_middle0, alllist0)
all_l1, all_r1, func1 = ListFormation(all_middle1, alllist1)

all_left0 = func0(all_l0)
all_right0 = func0(all_r0)

all_left1 = func1(all_l1)
all_right1 = func1(all_r1)


def RemoveDuplicates(seq):
    list7 = []
    for i in seq:
        if i not in list7:
            list7.append(i)
        else: continue
    return list7

def WithoutDup(seq1, seq2):
    withoutDupleft, withoutDupright = [], []
    for i in range(len(seq1)):
#         print(seq1[i])
#         print()
        g = RemoveDuplicates(seq1[i])
        withoutDupleft.append(g)

    for j in range(len(seq2)):
        h = RemoveDuplicates(seq2[j])
        withoutDupright.append(h)
    return withoutDupleft, withoutDupright

withoutDupleft0, withoutDupright0 = WithoutDup(all_left0, all_right0)
withoutDupleft1, withoutDupright1 = WithoutDup(all_left1, all_right1)    

def Removedollar(seq):
    ufinal = []
    for i in seq:
        final = []
        if i:
            for j in i:
                h = []
                a = len(j)
                if a > 1:
                    for k in j:
                        if k not in ['$', '(', ')']:
                            h.append(k)
                else:
                    if j not in ['$', '(', ')']:
                        h = j
                final.append(h)
            ufinal.append(final)
    return ufinal
withoutDupleft0 = Removedollar(withoutDupleft0)
# withoutDupright0 = Removedollar(withoutDupright0)
# withoutDupleft1 = Removedollar(withoutDupleft1)
# withoutDupright1 = Removedollar(withoutDupright0)
# withoutDupright0
all_l0


# In[98]:


from pprint import pprint
def Formation(withoutDupleft, withoutDupright, all_middle):
    all_lchunks, all_rchunks, all_mchunks, lchunks, rchunks = [], [], [], [], []
    
    for i in withoutDupleft:
        str1 = ''
        for j in i:
            b = len(j)
            if b > 1:
                for k in range(b):
                    str1 += '<' + j[k] + '>'
                str1 += '|'
            else:
                str1 += '<' + j[0] + '>' + '|'
        all_lchunks.append(str1[:-1])
#     print(all_lchunks)
    
    for i in withoutDupright:
        str2 = ''
        for j in i:
            b = len(j)
            if b > 1:
                for k in range(b):
                    str2 += '<' + j[k] + '>'
                str2 += '|'
            else:
                str2 += '<' + j[0] + '>' + '|'
        all_rchunks.append(str2[:-1])
#     print(all_rchunks)

    str3 = ''
    for i in all_middle:
        str3 = ''
        c = len(i)
        if c > 1:
            for j in range(c):
                str3 += '<' + i[j] + '>'
        else:
            str3 += '<' + i[0] + '>'
        all_mchunks.append(str3)
#     print(all_mchunks)    
    
    for i in range(len(all_mchunks)):
        z = ''  
        y = ''
        z += 'CHUNK' + str(i+1) + ':{(' + all_lchunks[i] + ')(' + all_mchunks[i] + ')}'
        y += 'CHUNK' + str(i+1) + ':{(' + all_mchunks[i] + ')(' + all_rchunks[i] + ')}'
        lchunks.append(z)
        rchunks.append(y)
    return lchunks, rchunks
    
nondef_chunks_l, nondef_chunks_r = Formation(withoutDupleft0, withoutDupright0, all_middle0)
def_chunks_l, def_chunks_r = Formation(withoutDupleft1, withoutDupright1, all_middle1)
def_chunks_l


# In[104]:


nondef_chunks_l


# In[204]:


cp1 = nltk.RegexpParser('CHUNK1 : {<NNP><DT><JJ><NNP><NN><IN><DT><NN><IN><DT><NNP><NNP><NNP><NNP><NNP><JJ><NN><NNS>}')
tagged_sentence0, tagged_sentence1 = [], []
for i in sent_0:
    text1 = []
    text = word_tokenize(i)
    for i in text:
        if i not in punclist:
            text1.append(i)
        else: continue
    tagged_0 = nltk.pos_tag(text1)
    tagged_sentence0.append(tagged_0)
    
for i in sent_1:
    text2 = []
    text = word_tokenize(i)
    for i in text:
        if i not in punclist:
            text2.append(i)
        else: continue
    tagged_1 = nltk.pos_tag(text2)
    tagged_sentence1.append(tagged_1)
    
total_tagged = tagged_sentence0 + tagged_sentence1
def GetSentence(tagged_doc):
    chunks = []
    cp1 = nltk.RegexpParser('CHUNK1 : {<IN><DT><JJ><NNP><NN><IN><DT><NN><IN><DT><NNP><NNP><NNP><NNP><NNP><CC>}')
    for i in tagged_doc:
        tree = cp1.parse(i)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK1':
                chunks.append(subtree.leaves())
    return chunks  

chunks = GetSentence(tagged_sentence0)
chunks


# In[31]:


all_lfreq0
all_middle0

def MakeFreqFrame(freqseq, all_mid, string):
    df7 = pd.DataFrame()
    for i in range(len(all_lfreq0)):
        f = list(freqseq[i].values())
        m = list(map(lambda x:list(x), list(freqseq[i].keys())))
        r = [all_mid[i]] * len(m)
        if string == 'l':
            p = {'freq':f, 'et':m, 'efe':r}
        else:
            p = {'freq':f, 'et':r, 'efe':m}
        t= pd.DataFrame(p)
        df7 = df7.append(t)
#     df7['combined'] = df7.apply(lambda x: list([x['et'], x['efe']]),axis=1)
#     df7['combined'] = df7['combined'].apply(lambda x: list(itertools.chain(*x)))
#     df7.drop(['et', 'efe'], axis =1, inplace = True)
    return df7  
leftfreqframe = MakeFreqFrame(all_lfreq0, all_middle0, 'l')
rightfreqframe = MakeFreqFrame(all_rfreq0, all_middle0, 'r')
leftfreqframe


# In[170]:


import gensim
from gensim.models import Word2Vec
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]
model = Word2Vec(sentences, min_count=1, window =1)
# print(model)
words = list(model.wv.vocab)
# print(words)
# print(model['sentence'])
# model.save('C:\\Users\jay.timbadia\Pictures\Saved Pictures\model.bin')
new_model = Word2Vec.load('C:\\Users\jay.timbadia\Pictures\Saved Pictures\crawl-300d-2M.vec')
new_model.wv['this']
# getlen = len(list(new_model.wv.vocab))
# list1 = []
# for i in range(getlen):
#     list1.append(new_model.wv[words[i]])
# len(list1)

# sentence = 'the merger was about to happen in New york dosent matter there was financial crisis'.split()
# from gensim.test.utils import datapath
# from gensim.models.word2vec import Text8Corpus
# from gensim.models.phrases import Phrases, Phraser
# # sentences = Text8Corpus(datapath('testcorpus.txt'))
# phrases = Phrases(sentence, min_count=1, threshold=1)  
# bigram = Phraser(phrases)  
# # for sent in bigram[sentence]:  
# #     print(sent)
    
    
# from gensim.models import Phrases
# documents = ["the mayor of new york was there financial crisis", "machine learning can be useful sometimes","new york mayor was present"]

# sentence_stream = [doc.split(" ") for doc in documents]
# bigram = Phrases(sentence_stream, min_count=1, threshold=2)
# sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there', u'financial', u'crisis']
# print(bigram[sent])

