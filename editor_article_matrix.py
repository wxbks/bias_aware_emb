#!/usr/bin/python
# encoding=utf8
from unidecode import unidecode

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_selection import chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy
from sklearn.cluster import KMeans
import json

def editor_articleNameLst(fileLst_, dic_):
    '''
       input: file list of train and dev 
       output: dict of editor-article
       note: only look at lines 4th column is true 
    '''
    dic = {}
    for f in fileLst_:
        with open(f, 'r') as a:
            for line in a:
                line = line.strip('\n')
                line = line.split('\t')
                if line[3] == 'true' and line[4] in dic:
                    if line[0] not in dic[line[4]]:
                        dic[line[4]].append(line[0])
                elif line[3] == 'true' and line[4] not in dic:
                    dic[line[4]] = [ line[0] ]
    print dic
    with open(dic_, 'w') as b:
        json.dump(dic, b)

def editor_editSentWord(fileLst_, dic_):
    dic = {}
    for f in fileLst_:
        with open(f, 'r') as a:
            for line in a:
                line = line.strip('\n')
                line = line.split('\t')
                if line[3] == 'true' and line[4] in dic:
                    dic[line[4]].append(line[6])
                elif line[3] == 'true' and line[4] not in dic:
                    dic[line[4]] = [ line[6] ]

    print dic
    with open(dic_, 'w') as b:
        json.dump(dic, b)

def editor_wordsLst(json_, dic_):
    '''
       input: dict json file of author: article list
       output: dict of editor-words
    '''
    au_word = {}
    le = WordNetLemmatizer()
    with open(json_) as a:
        art = json.load(a)
    for key, val in art.iteritems():
        
        temp = []
        for v in val:
            # preprocess
            vs = v.split()
            vs = [i.strip(string.punctuation) for i in vs]           # remove punc
            vs = [i.lower() for i in vs]
            vs = [le.lemmatize(i) for i in vs]
            temp = temp + vs
        temp = list(set(temp))
        au_word[key] = temp
    
    print au_word
    with open(dic_, 'w') as b:
        json.dump(au_word, b)


def editor_articleMatrixTransformed(art_):
    with open(art_) as a:
        art = json.load(a)
    art_name = []
    for key, val in art.iteritems():
        for i in val:
            if i not in art_name:
                art_name.append(i)
    artlen = len(art_name)
    artNum = range(artlen)
    art_dict = dict(zip(art_name, artNum))
    newart = []
    # integer encode input data
    new_dict = {}
    authors = [] 
    artOnly = []
    for i,v in art.iteritems():
        authors.append(i)
        s = [art_dict[u] for u in v]
        new_dict[i] = s
        artOnly.append(s)
    
    # one-hot encode categorical features
    onehot_encoded = list()
    for v in artOnly:
        temp = [0] * artlen
        for i in v:
            temp[i] = 1
            
        onehot_encoded.append(temp)

    return [authors, onehot_encoded]

def k_means(X):
    kmeans = KMeans(n_clusters=20)
    kmeans.fit(X)
    y_means = kmeans.predict(X)
    return y_means
    
def chi2_biasedWord(x_, y_):
    res = chi2(x_,y_)
    return res

def gen_chi2(auth_titleBeforeString_, auth_community_, chi2_):
    '''
       input: dict of author as key, split and processed title and biased before form as value
       output: chi2 weights
    '''
    with open(auth_titleBeforeString_) as a:
        auth_tb = json.load(a)
    with open(auth_community_) as b:
        auth_community = json.load(b)

    auth_com = dict(auth_community)
    x = []
    lab = []
    for key,val in auth_tb.iteritems():
        x.append(' '.join(val))
        lab.append(auth_com[key])
    vectorizer = TfidfVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(x)
    vocab = np.array(vectorizer.get_feature_names())
    dtm=dtm.toarray()
    keyness, _ = chi2(dtm, lab)
    ranking=np.argsort(keyness)[::-1]
    print vocab[ranking][:200]
    chi2Dict = dict(zip(vocab, keyness))
    with open(chi2_, 'w') as c:
        json.dump(chi2Dict, c)
    
def gen_author_splitTitleArtContent(artNamejson_, artContent_, out_):
    with open(artNamejson_) as a:
        artName = json.load(a)
    with open(artContent_) as b:
        artContent = json.load(b)
    
    if set(artName.keys()) == set(artContent.keys()):
        print 'equal'
    print artContent
    for key, val in artName.iteritems():
        
        if key in artContent:
            artContent[key].append(val[0])
            
    
    print artContent
    with open(out_, 'w') as c:
        json.dump(artContent, c)
    
    
def gen_author_SplitProcessWordTitCont(combine_, out_):
    '''
       input: dict of author as key and list contains title, article content as val
       output: dict of author as key and processed split words as val
    '''
    le = WordNetLemmatizer()
    with open(combine_) as a:
        combine = json.load(a)
    new = {}
    # preprocess
    for key, val in combine.iteritems():
        temp = []
        for v in val:
            vs = v.split()
            vs = [i.strip(string.punctuation) for i in vs]
            vs = [i.lower() for i in vs]
            vs = [le.lemmatize(i) for i in vs]
            temp = temp + vs
        temp = list(set(temp))
        new[key] = temp
    
    with open(out_, 'w') as b:
        json.dump(new, b)

def groupCluster(groupInfo_):
    with open(groupInfo_) as a:
        groupInfo = json.load(a)
    groupInfo = dict(groupInfo)
    v=defaultdict(list)
    for key,val in sorted(groupInfo.iteritems()):
        v[val].append(key)
    for i,j in v.iteritems():
        print i,j
        print "***********************************"


def stats_Data(train_eval_data_, train_eval_fileLst_, test_): 
    # stats of biased before form (phrases)
    # wiki_dic_editor_splitProcCombineTitleContentLst.json
    v_train_eval_phrases = []
    v_train_eval_sents = []
    v_train_eval_test_sents = []
    le = WordNetLemmatizer()    
    with open(train_eval_data_) as a:
        train_eval = json.load(a)

    for a,w in train_eval.iteritems():
        # filter stop words
        w = [v for v in w if v not in stopwords.words('english')]
        v_train_eval_phrases = v_train_eval_phrases + w
    # pre-process
    v_train_eval_phrases = [i.strip(string.punctuation) for i in v_train_eval_phrases]           # remove punc
    v_train_eval_phrases = [i.lower() for i in v_train_eval_phrases]
    v_train_eval_phrases = [le.lemmatize(i) for i in v_train_eval_phrases]
    v_train_eval_phrases = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', i) for i in v_train_eval_phrases]
    v_train_eval_phrases = filter(None, v_train_eval_phrases)
    

    print('v_train_eval_phrases token number:', len(v_train_eval_phrases))
    print('v_train_eval_phrases vocab number:', len(set(v_train_eval_phrases)))
    # stats of train eval sent vocab and token
    dic = {}
    for f in train_eval_fileLst_:
        with open(f, 'r') as a:
            for line in a:
                line = line.strip('\n')
                line = line.split('\t')
                if line[3] == 'true' and line[4] in dic:
                    if line[0] not in dic[line[4]]:
                        dic[line[4]].append(line[8])
                elif line[3] == 'true' and line[4] not in dic:
                    dic[line[4]] = [ line[8] ]
    # process sents

    for k,v in dic.iteritems():
        temp = []
        
        for i in v:
            i = process_modiString(i)
            temp.append(i)
        dic[k] = temp
        
    # split sents to words 
    for k,v in dic.iteritems():
        temp = []
        for t in v:
            
            vs = re.split(r'[\s|=]',t)
            vs = [i.strip(string.punctuation) for i in vs]           # remove punc
            vs = [i.lower() for i in vs]
            vs = [le.lemmatize(i.decode('utf-8')) for i in vs]
            vs = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', i) for i in vs]
            vs = filter(None, vs)
            temp = temp + vs
        temp = list(set(temp))
        dic[k]=temp
    for k, w in dic.iteritems():
        w = [v for v in w if v not in stopwords.words('english')]
        v_train_eval_sents = v_train_eval_sents + w
    v_train_eval_sents = v_train_eval_sents + v_train_eval_phrases
    print('v_train_eval_sents token number:', len(v_train_eval_sents))
    print('v_train_eval_sents vocab number:', len(set(v_train_eval_sents)))
    
    # stats of train eval test
    testWords = []
    dic_test = {}
    with open(test_) as b:
        for line in b:
            line = line.strip('\n')
            line = line.split('\t')
            if line[3] == 'true' and line[4] in dic_test:
                if line[0] not in dic_test[line[4]]:
                    dic_test[line[4]].append(line[8])
            elif line[3] == 'true' and line[4] not in dic_test:
                dic_test[line[4]] = [ line[8] ]
    for k,v in dic_test.iteritems():
        temp = []
        for i in v:
            i = process_modiString(i)
            temp.append(i)
        dic_test[k] = temp
    for k,v in dic_test.iteritems():
        temp = []
        for t in v:
            vs = re.split(r'[\s|=]',t)
            vs = [i.strip(string.punctuation) for i in vs]           # remove punc
            vs = [i.lower() for i in vs]
            vs = [le.lemmatize(i.decode('utf-8')) for i in vs]
            # strip html
            vs = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', i) for i in vs]
            vs = filter(None, vs)
            temp = temp + vs
        temp = list(set(temp))
        dic_test[k]=temp
    for k, w in dic_test.iteritems():
        w = [v for v in w if v not in stopwords.words('english')]
        testWords = testWords + w
    print('test words size: ', len(testWords))
    # print v_train_eval_sents
    
    v_train_eval_test_sents =  v_train_eval_sents + testWords
    print('v_train_eval_sents token number:', len(v_train_eval_test_sents))
    print('v_train_eval_sents vocab number:', len(list(set(v_train_eval_test_sents))))

# def splitProcessedString(dic_):
    # for k,v in dic_.iteritems():
        # temp = []
        # for t in 
         
def strippedNoSquBrac(test_str):
    ret = ''
    # skip1c = 0 #[                                                                  
    skip2c = 0 #<                                                                    
    skip3c = 0 #{                                                                    
    for i in test_str:
        if i == '<':
            skip2c += 1
        elif i == '{':
            skip3c += 1
        elif i == '>'and skip2c > 0:
            skip2c -= 1
        elif i == '>' and skip2c == 0:
            continue
        elif i == '}' and skip3c > 0:
            skip3c -= 1
        elif i == '}' and skip3c == 0:
            continue
        elif skip2c == 0 and skip3c == 0:
            ret += i
    return ret

def squrBracParse(str):
    if '[' in str and ']' in str:

        slist = re.findall('\[.*?\]\]?',str)


    elif '[' in str and ']' not in str:
        slist = re.findall('\[.*',str)
    else:
        return str

    for ins, sl in enumerate(slist):
        if '|' in sl:

            res1 = sl.split('|')
            if '-[' in str:
                        str = str.replace(sl,res1[-1].strip(']'))
            else:
                str = str.replace(sl,' '+res1[-1].strip(']'))
        else:

            nsl = sl.strip('[]')
            str = str.replace(sl,' ' +nsl)

    return str


## get rid of <>, {}; process [] and [|]
def process_modiString(str_):
    s1 = strippedNoSquBrac(str_)
    s2 = squrBracParse(s1)
    return s2
        
def processSentWiki(str_):
    # 1. remove html link in str_
    str_1  = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str_)
    # 2. remove html tags
    str_2 = BeautifulSoup(str_1).text
    # 3. process <>, [], {}, [|]
    str_3 = process_modiString(str_2)

if __name__ == "__main__":
    # editor_articleNameLst(['/home/sik211/dusk/npov_data/npov-edits/5gram-edits-train.tsv', '/home/sik211/dusk/npov_data/npov-edits/5gram-edits-dev.tsv'], 'wiki_editor_articleName_dict_4colTrue.json')
    numpy.set_printoptions(threshold=numpy.nan)
    
    # gen_author_SplitProcessWordTitCont('wiki_editor_combineTileContentList.json', 'wiki_dic_editor_splitProcCombineTitleContentLst.json')

    # gen_author_splitTitleArtContent('wiki_editor_articleName_dict_4colTrue.json', 'wiki_editor_articleBeforeForm.json', 'wiki_editor_combineTileContentList.json')
        
    [author, artLst] = editor_articleMatrixTransformed('wiki_dic_editor_splitProcCombineTitleContentLst.json')
    # apply PCA to reduce 10,000 dimen of artLst
    print len(artLst)
    print len(artLst[0])
    
    # print y
    # pca = PCA(n_components=10)
    # pca.fit(artLst)


    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    artLst1 = lsa.fit_transform(artLst)
    
    print len(artLst1)
    print len(artLst1[0])
    y = k_means(artLst1)
    # stats of y
    print Counter(y).keys() # equals to list(set(y))
    print Counter(y).values() # counts elements' frequency
    
    # record cluster's content
    # print y
    auth_clus = []

    for i,j in zip(author, y):
        auth_clus.append([i,int(j)])

    # print auth_clus
    # with open("wiki_author_communityGroup_20group.json", "w") as c:
        # json.dump(auth_clus, c)
    

    # print artLst
    # r = chi2_biasedWord(artLst, y)
    # print r
    
    # editor_wordsLst('wiki_editor_articleName_dict_4colTrue.json', 'wiki_editor_articleSplitWord_dict_4colTrue.json')

    # editor_editSentWord(['/home/sik211/dusk/npov_data/npov-edits/5gram-edits-train.tsv', '/home/sik211/dusk/npov_data/npov-edits/5gram-edits-dev.tsv'], 'wiki_editor_articleBeforeForm.json')
    # editor_wordsLst('wiki_editor_articleBeforeForm.json', 'wiki_editor_articleSplitWords.json')
    # gen_chi2('wiki_dic_editor_splitProcCombineTitleContentLst.json', 
             # 'wiki_author_communityGroup_20group.json', 'wiki_dict_chi2_biasedWord_20group.json')
    # groupCluster('wiki_author_communityGroup_20group.json')
    # stats_Data("wiki_dic_editor_splitProcCombineTitleContentLst.json",
               # ['/home/sik211/dusk/npov_data/npov-edits/5gram-edits-train.tsv', '/home/sik211/dusk/npov_data/npov-edits/5gram-edits-dev.tsv'], 
               # '/home/sik211/dusk/npov_data/npov-edits/5gram-edits-test.tsv')
