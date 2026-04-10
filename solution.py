from src.utils import load_data , preprocess_df , impute_missing_financials
from src.bm25 import BM25
import json
from src.cascade_clustering import run_cascade
import os

import time
from src.semantic_engine import SemanticEngine

from src.smart_hybrid_engine import create_union_pool , apply_soft_filters_and_rank

from src.interface import SearchInterface

# --- CONFIGURATION AREA ---
CONFIG = {
    'raw_dataset_path': "data/companies.jsonl",
    'clean_dataset_path': "data/companies_cleaned.jsonl",
    'output_path_stage0': "ranking_stage0.json",
    'output_path_stage1': "ranking_stage1.json",
    'output_path_stage2': "ranking_final.txt",
    'embeddings_path': "data/embeddings.npy",
    'semantic_companies_path': "data/semantic_companies.json",
    'top_k_stage1': 20,
    'top_k': 200,
    'bm25_k1': 1.5,
    'bm25_b': 0.75
}

def run_stage_0( query ) :
    print( f"Start searching engine for : '{ query }' . ")

    try :

        # Making a copy with cleaned data
        if not os.path.exists( CONFIG['clean_dataset_path'] ) :
            print( f"[ Init ] Generating cleaned copy in '{ CONFIG[ 'clean_dataset_path' ] }' ... " )
            impute_missing_financials( CONFIG[ 'raw_dataset_path' ] , CONFIG[ 'clean_dataset_path' ] )
        else :
            print( f"[ Init ] The cleaned file already exists in '{ CONFIG[ 'clean_dataset_path' ] }' ... " )

        raw_df = load_data( CONFIG['raw_dataset_path'] )
        original_data_df = raw_df.copy( deep = True )
        search_df = preprocess_df( raw_df )
        print( f"Uploaded and cleaned { len( search_df ) } companies .")

        corpus = []

        for _ , row in search_df.iterrows() :
            full_text = " ".join( str( val ) for val in row.values )
            corpus.append( full_text.lower().split() )

        print( "Calculating IDF and initializing BM25 ... ")
        bm25_engine = BM25(
            corpus ,
            k1 = CONFIG['bm25_k1'],
            b = CONFIG['bm25_b']
        )

        results = bm25_engine.rank_documents( query , top_n = CONFIG['top_k'] )
        print( "\nTop results BM25 : ")
        print( "-" * 60 )

        final_list = []
        for i , ( idx , score ) in enumerate( results , 1 ) :
            company = original_data_df.iloc[ idx ].to_dict()
            company[ "bm25_score" ] = score
            print( f"{ i } | { score } | { company.get( 'operational_name' , 'Unknown' ) }" )

            print( json.dumps( company , indent = 4 , ensure_ascii = False , default = str ) )
            print( "-" * 80 )

            final_list.append( company )

        # Cascade_clustering

        refined_list = run_cascade( query , final_list )

        # Print and verify results after filtering
        print( f"\n[ Final stage 0 ] : companies left - { len( refined_list ) } from top { CONFIG['top_k'] } BM25 " )
        for i , comp in enumerate( refined_list , 1 ) :
            addr = str( comp.get( 'address' , '' ) )
            print( f"{ i } | Score : { comp[ 'bm25_score' ]:.4f} | { comp.get( 'operational_name') } | Address : { addr } ..." )

        with open( CONFIG['output_path_stage0'] , 'w' , encoding = "utf-8" ) as f :
            json.dump( refined_list , f , indent = 4 , default = str )

        print( f"The results were saved in '{ CONFIG['output_path_stage0'] }' ." )
        return refined_list

    except Exception as e :
        print( f"Error in execution of solution.py : { e }" )



def run_stage_1( query ) :

    print( f"\n{ '=' * 70 }" )
    print( f"START STAGE 1 ( SEMANTIC SEARCH / AI EMBEDDINGS )" )

    engine = SemanticEngine()

    if not os.path.exists( CONFIG['embeddings_path'] ) or not os.path.exists( CONFIG['semantic_companies_path'] ) :
        print( "\n[ Offline phase ] The vectorial files are missing ... Generating them now ..." )

        if not os.path.exists( CONFIG['clean_dataset_path'] ) :
            print( "[ INIT ] The cleaned file are missing ... Generating it now using util file ..." )
            impute_missing_financials( CONFIG['raw_dataset_path'] , CONFIG['clean_dataset_path'] )

        engine.build_and_save_embeddings(
            dataset_path = CONFIG[ 'clean_dataset_path' ] ,
            save_path = CONFIG[ 'embeddings_path' ]
        )

    load_start = time.time()

    engine.load_embeddings(
        emb_path = CONFIG[ 'embeddings_path' ] ,
        comp_path = CONFIG[ 'semantic_companies_path' ]
    )
    print( f"[ File loading speed ] The upload from the disk took { time.time() - load_start:.4f} seconds." )

    search_start = time.time()
    print( "\n[ Searching ] Calculating the semantic distances in 3D space ..." )
    results = engine.search(
        query = query ,
        top_k = CONFIG[ 'top_k_stage1' ] ,
        min_score_threshold = 0.4
    )
    search_end = time.time()

    print( f"\nTop { CONFIG[ 'top_k_stage1' ] } results - STAGE 1 : " )
    print( "-" * 70 )
    for i , comp in enumerate( results , 1 ) :
        name = comp.get( 'operational_name' , 'Unknown' )
        score = comp.get( 'semantic_score' , 0.0 )
        desc = str( comp.get( 'description' , '' ) )[:90] + "..."

        print(f"{i:2d} | Scor Semantic: {score:.4f} | {name}")
        print(f"     Info: {desc}\n")

    with open( CONFIG[ 'output_path_stage1' ] , 'w' , encoding = "utf-8" ) as f :
        json.dump( results , f , indent = 4 , default = str )

    print( f"\n[ Speed ] Effective searching took { search_end - search_start :.4f} seconds." )
    print( f"Results from STAGE 1 were saved in { CONFIG[ 'output_path_stage1' ] }." )

    return results




def run_stage_2( query , stage0_results , stage1_results ) :
    print( f"\n{ '=' * 70 }" )
    print( f"Start Stage 2 ( Smart funnel / Hybrid engine )" )

    union_pool = create_union_pool( stage0_results , stage1_results )
    final_ranked_companies = apply_soft_filters_and_rank( query , union_pool )
    top_10 = final_ranked_companies[:10]

    output_path = CONFIG[ 'output_path_stage2' ]
    with open( output_path, 'w' , encoding = "utf-8" ) as f :
        f.write( f"Searched query : { query }\n" )
        f.write( "=" * 80 + "\n" )
        f.write( "Top 10 ranking :\n" )
        f.write( "=" * 80 + "\n" )

        for i , comp in enumerate( top_10 , 1 ) :
            name = comp.get( 'operational_name' , 'Unknown' )
            score = comp.get( 'hybrid_final_score' , 0.0 )
            logs = comp.get( 'soft_filter_note' , '' )
            f.write( f"{i:2d} | Final score : { score:.4f} | { name }\n")
            f.write( f"     [ Note ]: {logs}\n")

        f.write( "\n\n" + "=" * 80 + "\n" )
        f.write( "Complete details for the companies\n" )
        f.write( "=" * 80 + "\n\n" )

        for i , comp in enumerate( top_10 , 1 ) :
            f.write( f"Place { i } : { comp.get( 'operational_name' , 'Unknown' ) } \n")
            json_str = json.dumps( comp , indent = 4 , ensure_ascii = False , default = str )
            f.write( json_str )
            f.write( "\n\n" + "-" * 80 + "\n\n" )

    print( f"\n[ Success ] Pipeline completed ." )
    print( f"[ Success ] Project Done . Final file with top 10 companies and their details was saved in : '{ output_path }' " )

    return top_10




if __name__ == "__main__" :

    def execute_search_pipeline( query ) :

        stage0_res = run_stage_0( query )
        stage1_res = run_stage_1( query )
        top_10 = run_stage_2( query , stage0_res , stage1_res )

        return CONFIG[ 'output_path_stage2' ] , top_10

    ui = SearchInterface( search_callback = execute_search_pipeline )
    ui.run( port = 5000 )
