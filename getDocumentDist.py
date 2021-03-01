import nltk
import nltk.data
import os
from dataset.ISEAR.py_isear.isear_loader import IsearLoader
import torch
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import LdaModel
from gensim import corpora
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    mystopwords.append(w)

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    



attributes = ['EMOT','SIT']
target = ['TROPHO','TEMPER']
loader = IsearLoader(attributes, target, True)
data = loader.load_isear('./dataset/ISEAR/Isear.csv')

target=data.get_data()
text=data.get_freetext_content()



word2idx=torch.load('./data/wordIdx.pt')['word2idx']
idx2word=torch.load('./data/wordIdx.pt')['idx2word']

import random
random.seed(0)
N = range(0,len(text))
trainIdx = random.sample(N, int(0.6*len(text)))
testIdx = [i for i in range(0,len(text)) if i not in trainIdx]

trainSet=[]
testSet=[]

corpus=text
parser = StanfordCoreNLP(r'D:\stanfordParser\stanford-corenlp-full-2018-02-27')
corpus=[parser.word_tokenize(s) for s in corpus]
newcorpus=[]
for (index,sentence) in enumerate(corpus):
    newcorpus.append([])
    for word in sentence:
        if word not in mystopwords:
            newcorpus[index].append(word)
corpus=newcorpus
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(s) for s in corpus]
from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
    minimum_probability=0.0
)
documentTopicDis=torch.zeros(len(text),num_topics)
for (index,document) in enumerate(corpus):
    a=model[document]
    tops, probs = zip(*a)
    documentTopicDis[index]=torch.tensor(probs)
model.save("./data/ldaModel")
torch.save(documentTopicDis,"./data/documentTopicDis.pt")