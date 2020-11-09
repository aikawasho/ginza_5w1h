import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import numpy as np
import networkx as nx
import urllib.request, urllib.error
import math
import spacy
import operator
from parse_5w1h import parse_5w1h

#ストップワーズの読み込み
slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
slothlib_file = urllib.request.urlopen(slothlib_path)
slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

POS_KEPT = ["ADJ", "NOUN", "PROPN", "VERB"]
nlp = spacy.load('ja_ginza')

# コサイン類似度の計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def increment_edge (graph, node0, node1):
    
    if graph.has_edge(node0, node1):
        graph[node0][node1]["weight"] += 1.0
    else:
        graph.add_edge(node0, node1, weight=1.0)
        
#USE文ごと
def USE_sent(text):
    doc = nlp(text)
    sent_list = []
    sent_vectors = []
    summary = ''
    graph = nx.Graph()
    node_id = 0
    for sent in doc.sents:
        if sent.text not in sent_list:
            sent_list.append(sent.text)
            graph.add_node(node_id)
            vector = np.ravel(np.array(use(sent.text)))
            sent_vectors.append(vector)
            node_id += 1
    for i, vector1 in enumerate(sent_vectors[:-2]):
        for l,vector2 in enumerate(sent_vectors[i+1:],i+1):
            if cos_sim(vector1.T,vector2.T) > 0.4:
                increment_edge(graph, i, l)
    ranks = nx.pagerank(graph)
    for node_id, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:2]:
        summary = summary+sent_list[node_id]
    return summary

def link_sentence (doc, sent, lemma_graph, seen_lemma):
    visited_tokens = []
    visited_nodes = []

    for i in range(sent.start, sent.end):
        token = doc[i]

        if token.pos_ in POS_KEPT and token.lemma_ not in slothlib_stopwords:
            #token.lemma_は原型. token.pos_は品詞
            key = (token.lemma_, token.pos_)
            
            if key not in seen_lemma:
                seen_lemma[key] = set([token.i])
            else:
                seen_lemma[key].add(token.i)

            node_id = list(seen_lemma.keys()).index(key)

            if not node_id in lemma_graph:
                lemma_graph.add_node(node_id)

            #print("visit {} {}".format(visited_tokens, visited_nodes))
           # print("range {}".format(list(range(len(visited_tokens) - 1, -1, -1))))
            
            for prev_token in range(len(visited_tokens) - 1, -1, -1):
                #print("prev_tok {} {}".format(prev_token, (token.i - visited_tokens[prev_token])))
                
                if (token.i - visited_tokens[prev_token]) <= 3:
                    increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
                else:
                    break

            #print(" -- {} {} {} {} {} {}".format(token.i, token.text, token.lemma_, token.pos_, visited_tokens, visited_nodes))

            visited_tokens.append(token.i)
            visited_nodes.append(node_id)

def collect_phrases (chunk, phrases, counts,seen_lemma,ranks,doc):
    chunk_len = chunk.end - chunk.start + 1
    sq_sum_rank = 0.0
    non_lemma = 0
    compound_key = set([])

    for i in range(chunk.start, chunk.end):
        if i < len(doc):
            token = doc[i]
            key = (token.lemma_, token.pos_)
  
            if key in seen_lemma:
                node_id = list(seen_lemma.keys()).index(key)
                rank = ranks[node_id]
                sq_sum_rank += rank
                compound_key.add(key)

               # print(" {} {} {} {}".format(token.lemma_, token.pos_, node_id, rank))
            else:
                non_lemma += 1
    
    # although the noun chunking is greedy, we discount the ranks using a
    # point estimate based on the number of non-lemma tokens within a phrase
    non_lemma_discount = chunk_len / (chunk_len + (2.0 * non_lemma) + 1.0)
    # use root mean square (RMS) to normalize the contributions of all the tokens
    phrase_rank = math.sqrt(sq_sum_rank / (chunk_len + non_lemma))
    phrase_rank *= non_lemma_discount
    # remove spurious punctuation
    phrase = chunk.text.lower().replace("'", "")

    # create a unique key for the the phrase based on its lemma components
    compound_key = tuple(sorted(list(compound_key)))
    
    if not compound_key in phrases:
        phrases[compound_key] = set([ (phrase, phrase_rank) ])
        counts[compound_key] = 1
    else:
        phrases[compound_key].add( (phrase, phrase_rank) )
        counts[compound_key] += 1

   # print("{} {} {} {} {} {}".format(phrase_rank, chunk.text, chunk.start, chunk.end, chunk_len, counts[compound_key]))

#隣接する単語動詞をリンクする手法文ごと
def pyrank_sent(text):
    #parse = parse_5w1h(0)
    doc = nlp(text)
    lemma_graph = nx.Graph()
    seen_lemma = {}
    for sent in doc.sents:
        link_sentence(doc, sent, lemma_graph, seen_lemma)
        #break # only test one sentence
    labels = {}
    keys = list(seen_lemma.keys())

    for i in range(len(seen_lemma)):
        labels[i] = keys[i][0].lower()
    ranks = nx.pagerank(lemma_graph)
        
    phrases = {}
    counts = {}
    for sent in doc.sents:
        collect_phrases(sent, phrases, counts,seen_lemma,ranks,doc)
    min_phrases = {}
    
    for compound_key, rank_tuples in phrases.items():
        l = list(rank_tuples)
        l.sort(key=operator.itemgetter(1), reverse=True)

        phrase, rank = l[0]
        count = counts[compound_key]
        
        min_phrases[phrase] = (rank, count)
    ans = ''
    for phrase, (rank, count) in sorted(min_phrases.items(), key=lambda x: x[1][0], reverse=True)[0:3]:
            ans = ans+ phrase
    return(ans)

def collect_5w1hphrases (chunk,chunk_start,chunk_end, phrases,counts,seen_lemma,ranks,doc):
    chunk_len = chunk_end - chunk_start + 1
    sq_sum_rank = 0.0
    non_lemma = 0
    compound_key = set([])

    for i in range(chunk_start, chunk_end):
        if i < len(doc):
            token = doc[i]
            key = (token.lemma_, token.pos_)
  
            if key in seen_lemma:
                node_id = list(seen_lemma.keys()).index(key)
                rank = ranks[node_id]
                sq_sum_rank += rank
                compound_key.add(key)

               # print(" {} {} {} {}".format(token.lemma_, token.pos_, node_id, rank))
            else:
                non_lemma += 1
    
    # although the noun chunking is greedy, we discount the ranks using a
    # point estimate based on the number of non-lemma tokens within a phrase
    non_lemma_discount = chunk_len / (chunk_len + (2.0 * non_lemma) + 1.0)

    # use root mean square (RMS) to normalize the contributions of all the tokens
    phrase_rank = math.sqrt(sq_sum_rank / (chunk_len + non_lemma))
    phrase_rank *= non_lemma_discount

    # remove spurious punctuation
    phrase = chunk.replace("'", "")

    # create a unique key for the the phrase based on its lemma components
    compound_key = tuple(sorted(list(compound_key)))
    
    if not compound_key in phrases:
        phrases[compound_key] = set([ (phrase, phrase_rank) ])
        counts[compound_key] = 1
    else:
        phrases[compound_key].add( (phrase, phrase_rank) )
        counts[compound_key] += 1

    #print("{} {} {} {} {} {}".format(phrase_rank, chunk, chunk_start, chunk_end, chunk_len, counts[compound_key]))
    
#隣接する単語動詞をリンクする手法の5w1h
def pyrank_5w1h(text):
    parse = parse_5w1h(0)
    parse.extract(text)
    _5w1h = parse.display_5w1h()
    doc = parse.doc
    
    lemma_graph = nx.Graph()
    seen_lemma = {}
    for sent in doc.sents:
        link_sentence(doc, sent, lemma_graph, seen_lemma)
        #break # only test one sentence
    labels = {}
    keys = list(seen_lemma.keys())

    for i in range(len(seen_lemma)):
        labels[i] = keys[i][0].lower()
    ranks = nx.pagerank(lemma_graph)
    phrases = {}
    counts = {}

    for i,chunk in enumerate(_5w1h):

         collect_5w1hphrases(chunk.phrase, chunk.start,chunk.end,phrases,counts,seen_lemma,ranks,doc)
    min_ranks = {}

    for compound_key, rank_tuples in phrases.items():
        l = list(rank_tuples)
        l.sort(key=operator.itemgetter(1), reverse=True)

        phrase, rank = l[0]

        min_ranks[phrase] = (rank)
    ans = ''
#####要約最新#####
    i = 0 
    summary = []
    count = []
    while i < len(_5w1h):
        if _5w1h[i].phrase in min_ranks:
            if min_ranks[_5w1h[i].phrase] > 0.040:
                if _5w1h[i].phrase not in count:
                    summary.append(_5w1h[i].phrase)
                    count.append(_5w1h[i].phrase)

                if _5w1h[i]._type =="How":
                    l = i-1
                    while l >= 0:
                        if _5w1h[l].phrase not in count:
                            summary.insert(0,_5w1h[l].phrase)
                            count.append(_5w1h[l].phrase)
                        if _5w1h[l]._type == 'Who'or _5w1h[l]._type == 'What':
                            if summary != []:
                                ans = ans + ''.join(summary)
                            summary = []
                            break
                        l -= 1
                else:
                    i = i+1
                    while i < len(_5w1h):
                        if _5w1h[i].phrase not in count:
                            summary.append(_5w1h[i].phrase)
                            count.append(_5w1h[i].phrase)
                        if i+1 < len(_5w1h):
                            if _5w1h[i]._type == 'How' and _5w1h[i+1].phrase in min_ranks:
                                    if min_ranks[_5w1h[i+1].phrase] < 0.040 and summary != []:
                                        ans = ans + ''.join(summary)
                                    summary = []
                                    break
                        i += 1

        i += 1
    return(ans)

#USE_5w1hごと
def USE_5w1h(text):
    parse = parse_5w1h(0)
    parse.extract(text)
    _5w1h = parse.display_5w1h()  
    summary = ''
    phrase_list = []
    _5w1h_vectors = []
    graph = nx.Graph()

    node_id = 0
    for phrase in _5w1h:
        if phrase.phrase not in phrase_list:
            phrase_list.append(phrase.phrase)
            graph.add_node(node_id)
            vector = np.ravel(np.array(use(phrase.phrase)))
            _5w1h_vectors.append(vector)
            node_id += 1
            
    #類似度が0以上の場合edgeで繋ぐ
    for i, vector1 in enumerate(_5w1h_vectors[:-2]):
        for l,vector2 in enumerate(_5w1h_vectors[i+1:],i+1):
            if cos_sim(vector1,vector2) > 0.85:
                increment_edge(graph, i, l)
                
    ranks = nx.pagerank(graph)
    min_ranks = {}
    for node_id, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
        min_ranks[phrase_list[node_id]] = rank
        
    ans = ''
#####要約最新#####
    i = 0 
    summary = []
    count = []
    while i < len(_5w1h):
        if _5w1h[i].phrase in min_ranks:
            if min_ranks[_5w1h[i].phrase] > 0.040:
                if _5w1h[i].phrase not in count:
                    summary.append(_5w1h[i].phrase)
                    count.append(_5w1h[i].phrase)

                if _5w1h[i]._type =="How":
                    l = i-1
                    while l >= 0:
                        if _5w1h[l].phrase not in count:
                            summary.insert(0,_5w1h[l].phrase)
                            count.append(_5w1h[l].phrase)
                        if _5w1h[l]._type == 'Who'or _5w1h[l]._type == 'What':
                            if summary != []:
                                ans = ans + ''.join(summary)
                            summary = []
                            break
                        l -= 1
                else:
                    i = i+1
                    while i < len(_5w1h):
                        if _5w1h[i].phrase not in count:
                            summary.append(_5w1h[i].phrase)
                            count.append(_5w1h[i].phrase)
                        if i+1 < len(_5w1h):
                            if _5w1h[i]._type == 'How' and _5w1h[i+1].phrase in min_ranks:
                                    if min_ranks[_5w1h[i+1].phrase] < 0.040 and summary != []:
                                        ans = ans + ''.join(summary)
                                    summary = []
                                    break
                        i += 1

        i += 1
    return(ans)

#公式に載っていた, 文ごとに抽出する手法
def pytext_sum(text):
    doc = nlp(text)
    sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]

    ans = ''
    
    limit_phrases = 4

    phrase_id = 0
    unit_vector = []

    for p in doc._.phrases:
        #print(phrase_id, p.text, p.rank)

        unit_vector.append(p.rank)

        for chunk in p.chunks:
            #print(" ", chunk.start, chunk.end)

            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.start <= sent_end:
                    #print(" ", sent_start, chunk.start, chunk.end, sent_end)
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break

    sum_ranks = sum(unit_vector)
    unit_vector = [ rank/sum_ranks for rank in unit_vector ]


    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        #print(sent_vector)
        sum_sq = 0.0

        for phrase_id in range(len(unit_vector)):
            #print(phrase_id, unit_vector[phrase_id])

            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0

        sent_rank[sent_id] = math.sqrt(sum_sq)
        sent_id += 1
        
    sorted(sent_rank.items(), key=operator.itemgetter(1))
    limit_sentences = 3

    sent_text = {}
    sent_id = 0

    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    num_sent = 0

    for sent_id, rank in sorted(sent_rank.items(), key=operator.itemgetter(1)):
        #print(sent_id, sent_text[sent_id])
        num_sent += 1
        ans = ans + sent_text[sent_id]
        if num_sent == limit_sentences:
            break
    return(ans)