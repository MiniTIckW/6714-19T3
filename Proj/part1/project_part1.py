## Import Libraries and Modules here...
import spacy
import math

class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = {}
        self.tf_entities = {}

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = {}
        self.idf_entities = {}

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        nlp = spacy.load("en_core_web_sm")
        token = {}
        entity = {}
        idf_tokens = {}
        idf_entities = {}
        l = len(documents)
        for i in documents:
            entities = [str(ett) for ett in nlp(documents[i]).ents if nlp.vocab[str(ett)].is_stop == False]
            doc = [token.orth_ for token in nlp(documents[i]) if token.is_punct == False and token.is_space == False and token.is_stop == False]                    
            for k in entities:
                if k not in entity:
                    entity[k] = {i:1}
                else:
                    if i not in entity[k]:
                        entity[k][i] = 1
                    else:
                        entity[k][i] += 1
            for j in doc:
                if j in entities:
                    n = doc.count(j) - entities.count(j)
                    if n != 0: 
                        if j not in token:
                            token[j] = {i:n}
                        else:
                            token[j][i] = n
                else:
                    if j not in token:
                        token[j] = {i:1}
                    else:
                        if i not in token[j]:
                            token[j][i] = 1
                        else:
                            token[j][i] += 1
        

        for p in token:
            idf_tokens[p] = 1 + math.log(l/(1+len(token[p])))
        for q in entity:
                idf_entities[q] = 1 + math.log(l/(1+len(entity[q])))
        
        self.tf_tokens = token
        self.tf_entities = entity
        self.idf_tokens = idf_tokens
        self.idf_entities = idf_entities

    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        nlp = spacy.load("en_core_web_sm")
        q = [token.orth_ for token in nlp(Q) if not token.is_punct | token.is_space]
        query = [[[],q]]
        newq = [[[],q]]

        while newq != []:
            left = []
            for [e,k] in newq:
                for i in DoE:
                    words = i.split()
                    l = len(words)
                    nb = 0
                    for j in range(l):
                        if words[j] in k:
                            nb += 1
                    if nb == l:
                        newk = k[:]
                        for w in words:
                            newk.remove(newk[newk.index(w)])
                        if e != []:
                            if Q.index(words[0]) > Q.index(e[-1].split()[0]):
                                newe = e[:]
                                newe.append(i)
                                query.append([newe,newk])
                                left.append([newe,newk])
                        else:
                            newe = e[:]
                            newe.append(i)
                            query.append([newe,newk])
                            left.append([newe,newk])

            newq = left

        return query
    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        s = 0
        max_score = 0
        for [e,k] in query_splits:
            s1 = 0
            s2 = 0
            if e != []:
                for i in e:
                    if i in self.tf_entities and doc_id in self.tf_entities[i]:
                        tf_e = self.tf_entities[i][doc_id]
                        tf_ne = 1 + math.log(tf_e)
                        idf = self.idf_entities[i]
                        s1 += idf * tf_ne
            if k != []:
                for j in k:
                    if j in self.tf_tokens and doc_id in self.tf_tokens[j]:
                        tf_k = self.tf_tokens[j][doc_id]
                        tf_nk = 1 + math.log(1 + math.log(tf_k))
                        idf = self.idf_tokens[j]
                        s2 += idf * tf_nk

            s = s1 + 0.4*s2
            if s > max_score:
                max_score = s
                tk = k
                et = e
        result = (max_score,{'tokens':tk,'entities':et})
        
        return result
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
