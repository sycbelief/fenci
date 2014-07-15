from gensim import corpora, models, similarities
_filename = ".txt"

texts = []
for i in range(5):
    filename = str(i+1) + _filename
    file = open(filename, 'r', encoding = "gbk")
    tmp = file.readline().split()
    texts.append(tmp)
print (texts)
print ()

dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')
print (dictionary.token2id)
print ()

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)
print (corpus)
print ()


tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

for doc in corpus_tfidf:
    print (doc)
