import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer , util

class SemanticEngine :
    def __init__( self , model_name = 'all-MiniLM-L6-v2' ) :
        """
        Initializing AI model
        """

        print( f"[ Semantic Engine ] Loading the neural model { model_name } ... " )
        self.model = SentenceTransformer( model_name )
        self.companies = []
        self.embeddings = None

    def _clean_query_for_ai( self , query ) :
        """
        Extracting only the business intent
        """

        clean_query = query.lower()
        clean_query = re.sub(r'\b(exclude|excluding|without|except|not|no)\b(\s+\w+){1,2}' , ' ' , clean_query )
        clean_query = re.sub(r'(more than|less than|under|over|founded after|founded before)\s*\d+\s*(employees|staff)?' , ' ' , clean_query )
        clean_query = re.sub(r'\b(19|20)\d{2}\b' , ' ' , clean_query )
        clean_query = re.sub(r'\s+' , ' ' , clean_query ).strip()

        print( f"[ AI LENS ] Original query : { query }" )
        print( f"[ AI LENS ] Cleaned query : { clean_query }" )

        return clean_query


    def _create_rich_text( self , comp ) :
        """
        V1 : Transforming the dictionary of the company into one long readable paragraph
        V2 : Focusing only on the business intent
        """

        naics = comp.get( 'primary_naics' , {} )
        industry = naics.get( 'label' , '' ) if isinstance( naics , dict ) else str( naics )
        biz_models = ", ".join( [ str( m ) for m in comp.get( 'business_model' , [] ) ] )
        target_markets = ", ".join( [ str( t ) for t in comp.get( 'target_markets' , [] ) ] )
        offerings = ", ".join( [ str( o ) for o in comp.get( 'core_offerings' , [] ) ][:8] )
        desc = comp.get( 'description' , '' )


        # Building the story
        text = f"Industry : { industry } . "
        text += f"Target Markets : { target_markets } . "
        text += f"Business Models : { biz_models } . "
        text += f"Core Services and Offerings : { offerings } . "
        text += f"Description : { desc } . "

        return text.replace('\n', ' ').replace('  ', ' ').strip()



    def build_and_save_embeddings( self , dataset_path , save_path = "data/embeddings.npy" ) :
        """
        Reads the companies , turns them into text , calculating the 3D coordinates ( embeddings ) and
        saves them on disk in order not to lose time for the next run .
        """

        print( "\n[ Semantic Engine ] Building the embeddings ( it may take some minutes ) ... " )

        self.companies = []
        texts_to_embed = []

        with open( dataset_path , 'r' , encoding = 'utf-8' ) as f :
            for line in f :
                if not line.strip() : continue
                comp = json.loads( line )
                self.companies.append( comp )

                rich_text = self._create_rich_text( comp )
                texts_to_embed.append( rich_text )

        self.embeddings = self.model.encode( texts_to_embed , convert_to_numpy = True , show_progress_bar = True )
        np.save( save_path , self.embeddings )

        with open( "data/semantic_companies.json" , 'w' , encoding = 'utf-8' ) as f :
            json.dump( self.companies , f , ensure_ascii = False )

        print( f"[ Semantic Engine ] Saved the coordinates for { len( self.companies ) } companies in { save_path } ." )



    def load_embeddings( self , emb_path="data/embeddings.npy" , comp_path="data/semantic_companies.json" ) :
        print( "[ Semantic Engine ] Loading the semantic data base from the disk ... " )
        self.embeddings = np.load( emb_path )
        with open( comp_path , 'r' , encoding = 'utf-8' ) as f :
            self.companies = json.load( f )




    def search( self , query , top_k = 10 , min_score_threshold = 0.5) :
        """
        Turns query into coordinates and calculates the distance to the companies using Cosine Similarity
        """

        if self.embeddings is None :
            raise ValueError( "Embeddings not loaded ! Call load_embeddings() first !" )

        pure_query = self._clean_query_for_ai( query )
        query_embedding = self.model.encode( pure_query , convert_to_numpy = True )
        cos_scores = util.cos_sim( query_embedding , self.embeddings )[0]

        k = min( top_k , len( cos_scores ) )

        top_results = np.argpartition( -cos_scores , range( k ) )[ :k ]
        top_results = top_results[ np.argsort( -cos_scores[ top_results ] ) ]
        results = []
        for idx in top_results :
            # Adding a minimum threshold for the scores
            score = float( cos_scores[ idx ] )
            if score < min_score_threshold :
                continue

            comp_copy = self.companies[ int( idx ) ].copy()
            comp_copy[ 'semantic_score' ] = float( cos_scores[ idx ] )
            results.append( comp_copy )

        return results



