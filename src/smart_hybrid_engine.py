import json
import re
import math
from src.constants import COUNTRIES_MAP
from sentence_transformers import CrossEncoder


# Loading the Cross-Encoder model globally
_cross_encoder_model = None

def get_cross_encoder() :
    global _cross_encoder_model
    if _cross_encoder_model is None :
        print( "[ Hybrid Engine ] Loading Cross-Encoder ( MS_MARCO ) for Deep Semantic Scoring ... " )
        _cross_encoder_model = CrossEncoder( 'cross-encoder/ms-marco-MiniLM-L-6-v2' , max_length = 512 )
    return _cross_encoder_model

def sigmoid( x ) :  # Using sigmoid function in order to normalize scores for a veto verification
     return 1 / ( 1 + math.exp( -x ) )


def create_union_pool( stage0_results , stage1_results ) :
    """
    Combining the results from both stages
    """

    pool = {}

    for comp in stage0_results :
        comp_id = comp.get( 'website' ) or str( comp.get( 'operational_name' , '' ) ).lower()
        if comp_id :
            pool[ comp_id ] = comp.copy()
            pool[ comp_id ][ 'source_stage0' ] = True
            pool[ comp_id ][ 'source_stage1' ] = False

    for comp in stage1_results :
        comp_id = comp.get( 'website' ) or str( comp.get( 'operational_name' , '' ) )
        if comp_id :
            if comp_id in pool :
                pool[ comp_id ][ 'source_stage1' ] = True
                pool[ comp_id ][ 'semantic_score' ] = comp.get( 'semantic_score' , 0 )
            else :
                pool[ comp_id ] = comp.copy()
                pool[ comp_id ][ 'source_stage0' ] = False
                pool[ comp_id ][ 'source_stage1' ] = True

    return list( pool.values() )



def apply_soft_filters_and_rank( query , union_pool ) :
    """
    This function represents the mathematical brain behind Stage 2
    It combines decay functions with Deep Semantic Scoring made by Cross-Encoder
    """

    if not union_pool :
        return []

    query_lower = query.lower()
    final_ranked_companies = []

    print( f"\n[ Stage 2 ] Evaluating Cross-Encoder for { len( union_pool ) } companies ... " )
    cross_model = get_cross_encoder()

    pairs = []
    for comp in union_pool :
        offerings = " ".join( [ str( o ) for o in comp.get( 'core_offerings' , [] ) ][:8] )
        desc = str( comp.get( 'description' , '' ) )[:300]
        doc_text = f"Offerings: {offerings}. Description: {desc}"
        pairs.append( [ query , doc_text ] )

    ce_scores_raw = cross_model.predict( pairs )
    # Normalizing the scores using Min-Max Scaling
    min_ce = float( min( ce_scores_raw ) )
    max_ce = float( max( ce_scores_raw ) )


    emp_match_under = re.search( r'(<|under|less than|fewer than)\s*(\d+)' , query_lower )
    emp_match_over = re.search( r'(>|over|more than|at least)\s*(\d+)' , query_lower )
    max_emp_target = int( emp_match_under.group( 2 ) ) if emp_match_under else float( 'inf' )
    min_emp_target = int( emp_match_over.group( 2 ) ) if emp_match_over else 0

    year_match_after = re.search( r'(after|>|since|post)\s*(\d{4})' , query_lower )
    year_match_before = re.search( r'(before|<|prior to|until|older than|earlier than)\s*(\d{4})' , query_lower )
    min_year_target = int( year_match_after.group( 2 ) ) if year_match_after else 0
    max_year_target = int( year_match_before.group( 2 ) ) if year_match_before else float( 'inf' )

    target_country_code = None
    for country , ( code , keywords ) in COUNTRIES_MAP.items() :
        if any( f" { kw } " in f" { query_lower } " for kw in keywords ) :
            target_country_code = code
            break



    # Calculating the mathematical final score for every company

    for idx , comp in enumerate( union_pool ) :

        logs = []

        #important : Semantic base score ( Bi-encoder )
        base_score = comp.get( 'semantic_score' , 0.4 )
        final_score = base_score

        #important : Deep semantic boost ( Cross-Encoder )
        if max_ce > min_ce :
            ce_confidence = ( float( ce_scores_raw[ idx ] ) - min_ce ) / ( max_ce - min_ce )
        else :
            ce_confidence = 0.0
        ce_boost = ce_confidence * 0.25
        final_score += ce_boost
        logs.append( f"+ { ce_boost:.2f} ( Deep Cross-Encoder )" )
        absolute_confidence = sigmoid( float( ce_scores_raw[ idx ] ) )
        if comp.get( 'source_stage0' ) and not comp.get( 'source_stage1' ) :
            if absolute_confidence < 0.15 :
                final_score -= 0.35
                logs.append( f"- 0.35 ( Semantic Veto : Irrelevant BM25 hit )" )

        #important : Dual-source bonus
        if comp.get( 'source_stage0' ) and comp.get( 'source_stage1' ) :
            final_score += 0.10
            logs.append( "+ 0.10 ( Dual Match )" )

        #important : Lexical Boost ( BM25 )
        bm25 = comp.get( 'bm25_score' , 0 )
        if bm25 > 0 :
            bm25_boost = min( 0.10 , bm25 * 0.005 )
            final_score += bm25_boost
            logs.append( f"+ { bm25_boost:.2f} ( Lexical BM25 )" )

        #important : Geo-Anchor Penalty
        if target_country_code :
            address_str = str( comp.get( 'address' , '' ) ).lower()
            if target_country_code not in address_str :
                final_score -= 0.20
                logs.append( "- 0.20 ( Geographic Mismatch )" )
            else :
                final_score += 0.05
                logs.append( "+ 0.05 ( Geographic Match )" )

        #important : Temporal Decay Function
        year = comp.get( 'year_founded' )
        has_year_rule = min_year_target > 0 or max_year_target != float( 'inf' )
        if has_year_rule :
            if year and str( year ).replace( '.0' , '' ).isdigit() :
                comp_year = int( float( year ) )

                if comp_year <= min_year_target and min_year_target > 0 :
                    years_off = min_year_target - comp_year
                    penalty = min( 0.25 , years_off * 0.015 )
                    final_score -= penalty
                    logs.append( f"- { penalty:.2f} ( Age Min Dev : { years_off }y too old )" )

                elif comp_year >= max_year_target and max_year_target != float( 'inf' ) :
                    years_off = comp_year - max_year_target
                    penalty = min( 0.25 , years_off * 0.015 )
                    final_score -= penalty
                    logs.append( f"- { penalty:.2f} ( Age Max Dev : { years_off }y too new )" )

            else :
                final_score -= 0.05
                logs.append( "- 0.05 ( Missing Year )" )

        #important : Employee Deviation Penalty
        emp_count = comp.get( 'employee_count' )
        has_emp_rule = max_emp_target != float( 'inf' ) or min_emp_target > 0

        if has_emp_rule :
            if emp_count and str( emp_count ).replace( '.0' , '' ).isdigit() :
                emp_val = int( float( emp_count ) )

                if emp_val > max_emp_target :
                    deviation_ratio = ( emp_val - max_emp_target ) / max_emp_target
                    penalty = min( 0.35 , deviation_ratio * 0.05 )
                    final_score -= penalty
                    logs.append( f"- { penalty:.2f} ( Emp Max Dev : { emp_val }" )

                elif emp_val < min_emp_target and min_emp_target > 0 :
                    deviation_ratio = ( min_emp_target - emp_val ) / min_emp_target
                    penalty = min( 0.35 , deviation_ratio * 0.35 )
                    final_score -= penalty
                    logs.append( f"- { penalty:.2f} ( Emp Min Dev : { emp_val }" )

            else :
                final_score -= 0.05
                logs.append( "- 0.05 ( Missing Emp Count )" )


        #important: Computing the final score
        comp[ 'hybrid_final_score' ] = final_score
        comp[ 'soft_filter_note' ] = " | ".join( logs )
        final_ranked_companies.append( comp )

    final_ranked_companies.sort( key = lambda x : x[ 'hybrid_final_score' ] , reverse = True )
    return final_ranked_companies




