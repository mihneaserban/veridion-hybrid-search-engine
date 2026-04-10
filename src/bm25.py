import math
from collections import Counter

class BM25 :
    def __init__( self , corpus , k1=1.5 , b=0.75 ) :
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        if self.n == 0:
            self.avgdl = 0
            self.doc_len = []
            self.doc_freqs = {}
            self.idf = {}
            return

        self.doc_len = [ len(doc) for doc in corpus ]
        total_len = sum( self.doc_len )
        if total_len == 0 :
            self.avgdl = 0
        else :
            self.avgdl = total_len / self.n

        self.doc_freqs = self._calculate_doc_freqs()
        self.idf = self._calculate_idf()

    def _calculate_doc_freqs( self ) :
        # count in how many documents each word appears ( document frequency )
        df = {}
        for doc in self.corpus :
            unique_terms = set( doc )
            for term in unique_terms :
                df[ term ] = df.get( term , 0 ) + 1
        return df

    def _calculate_idf( self ) :
        # calculate the idf ( inverse document frequency ) using the formula for BM25
        idf = {}
        for term , freq in self.doc_freqs.items() :
            val = math.log( ( self.n - freq + 0.5 ) / ( freq + 0.5 ) + 1.0 )
            idf[ term ] = val
        return idf

    def get_score( self , query_tokens , doc_index ) :

        if self.avgdl == 0 :
            return 0.0

        # calculate the score for one document using the formula from "BM25 - formula.png"
        score = 0.0
        doc = self.corpus[ doc_index ]
        doc_counter = Counter( doc )
        d_len = self.doc_len[ doc_index ]

        for term in query_tokens :
            if term not in self.idf :
                continue

            f_qi_D = doc_counter[ term ]
            term_idf = self.idf[ term ]

            # computing the denominator
            denominator = f_qi_D + self.k1 * ( 1 - self.b + self.b * ( d_len / self.avgdl ) )
            # computing the fraction
            if denominator > 0 :
                fraction = ( f_qi_D * ( self.k1 + 1 ) ) / denominator
                # computing the score
                score += term_idf * fraction

        return score

    def rank_documents( self , query , top_n = 10 ) :
        # This function gets a query which is tokenized and return the indexes and scores for the most relevant documents
        query_tokens = query.lower().split()
        scores = []

        for i in range( self.n ) :
            score = self.get_score( query_tokens , i )
            scores.append( ( i , score ) )

        return sorted( scores , key = lambda x : x[1] , reverse = True )[:top_n]
