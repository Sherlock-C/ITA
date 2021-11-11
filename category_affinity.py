
import gensim
from gensim import corpora, models
import re

def cal_venues():
    venues = {}
    count = 0
    with open('./venues', 'r') as f:
        for line in f.readlines():
            if count == 0:
                count += 1
                continue
            else:
                data = re.split("\t", line.strip())

                if data[0] not in venues.keys():
                    if len(data) == 1:
                        venues[data[0]] = 'NULL'
                    else:
                        venues[data[0]] = data[1]

    return venues

def cal_location_venue():

    venues = cal_venues()
    location_venues = {}
    count = 0
    with open('./location_venue', 'r') as f:
        for line in f.readlines():
            if count == 0:
                count += 1
                continue
            else:

                data = line.strip().split('\t')

                if data[0] not in location_venues.keys():
                    location_venues[data[0]] = list()
                    if len(data) == 1:
                        location_venues[data[0]] = 'NULL'
                    else:
                        location_venues[data[0]].append(venues[data[1]])
                else:
                    location_venues[data[0]].append(venues[data[1]])
    return location_venues


def get_cat_affi():

    location_venues = cal_location_venue()

    N = 11326


    user_with_cat = [0] * N

    file = open("./checkins")

    data = []

    b = []

    for line in file.readlines():
        data.append(line.strip())
    file.close()

    for row in range(N):
        b.append([])

    for i in range(len(data)):
        if i == 0:
            continue
        c = re.split('\t', data[i])

        if(len(c) == 5):
            if(c[2]!='0.0' and c[3] != '0.0'):
                if c[4] not in location_venues.keys():
                    continue
                else:
                    user_with_cat[int(c[0][1:])] = 1
                    for v in location_venues[c[4]]:
                        if v != 'NULL':
                            b[int(c[0][1:])].append(v)
                        else:
                            continue


    return b, user_with_cat, location_venues


def compute_lda():

    user_cat, user_with_cat, location_venues = get_cat_affi()
    processed_docs = []

    for i in range(len(user_with_cat)):
        if user_with_cat[i] == 0:
            continue
        else:
            processed_docs.append(user_cat[i])

    dictionary = gensim.corpora.Dictionary(processed_docs)

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)

    corpus_tfidf = tfidf[bow_corpus]

    lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=50,
                                           id2word=dictionary, passes=2, workers=4)

    return lda_model, dictionary, user_cat, user_with_cat, location_venues
