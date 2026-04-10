import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.constants import STOP_WORDS_QUERY , COUNTRIES_MAP


# First layer of filtering
# Geographic + temporal + employee_count

def geo_temporal_employee_filter( query , companies ) :

    query_lower = query.lower()
    filtered_companies = companies

    # Temporal Detection - English Expressions
    # Dividing the words into mathematical operators ( < , > , == )

    regex_next = r'(the year after|year after|next year after)\s*(\d{4})'
    regex_after = r'(after|post|since|newer than|later than|starting from|>)\s*(\d{4})'
    regex_before = r'(before|prior to|until|up to|older than|earlier than|<)\s*(\d{4})'
    regex_exact = r'(founded in|established in|in the year|for the year|in|from|year|==|=)\s*(\d{4})'

    target_year = None
    operator = None

    if re.search( regex_next , query_lower ) :
        match = re.search( regex_next , query_lower )
        target_year = int( match.group( 2 ) ) + 1
        operator = '=='
    elif re.search( regex_after , query_lower ) :
        match = re.search( regex_after , query_lower )
        target_year = int( match.group( 2 ) )
        operator = '>'
    elif re.search( regex_before , query_lower ) :
        match = re.search( regex_before , query_lower )
        target_year = int( match.group( 2 ) )
        operator = '<'
    elif re.search( regex_exact , query_lower ) :
        match = re.search( regex_exact , query_lower )
        target_year = int( match.group( 2 ) )
        operator = '=='

    if target_year and operator :
        print( f"[ Filter 1 ] activated ... Founded year { operator } { target_year }" )

        passed_companies = []
        for comp in filtered_companies :
            year = comp.get( 'year_founded' )
            if year and str( year ).replace( '.0' , '' ).isdigit() :
                comp_year = int( float( year ) )

                if operator == '>' and comp_year > target_year :
                    passed_companies.append( comp )
                elif operator == '<' and comp_year < target_year :
                    passed_companies.append( comp )
                elif operator == '==' and comp_year == target_year :
                    passed_companies.append( comp )

        filtered_companies = passed_companies

    # Employee filter

    regex_emp_over = r'(>|over|more than|at least|minimum)\s*(\d+)\s*(employees|people|staff)'
    regex_emp_under = r'(<|under|less than|fewer than|maximum|at most|up to)\s*(\d+)\s*(employees|people|staff)'
    match_emp_over = re.search( regex_emp_over , query_lower )
    match_emp_under = re.search( regex_emp_under , query_lower )

    if match_emp_over or match_emp_under :
        min_employees = int( match_emp_over.group( 2 ) ) if match_emp_over else 0
        max_employees = int( match_emp_under.group( 2 ) ) if match_emp_under else float( 'inf' )

        if match_emp_over and match_emp_under :
            print( f"[ Filter 1 ] activated ... Employees between { min_employees } and { max_employees } " )
        elif match_emp_over :
            print( f"[ Filter 1 ] activated ... Employees >= { min_employees } " )
        elif match_emp_under :
            print( f"[ Filter 1 ] activated ... Employees <= { max_employees } " )

        passed_companies = []
        for comp in filtered_companies :
            emp = comp.get( 'employee_count' )
            if emp and str( emp ).replace( '.0' , '' ).isdigit() :
                emp_val = int( float( emp ) )
                if min_employees <= emp_val <= max_employees :
                    passed_companies.append( comp )
        filtered_companies = passed_companies

    # Geographic detection

    target_country_name = None
    target_country_code = None
    for country , ( code , keywords ) in COUNTRIES_MAP.items() :
        search_terms = [ f" { country } " ] + [ f" { kw } " for kw in keywords ]
        padded_query = f" { query_lower } "
        if any( term in padded_query for term in search_terms ) :
            target_country_name = country
            target_country_code = code
            break

    if target_country_name :
        print( f"[ Filter 1 ] activated ... Found country { target_country_name.upper() } with code { target_country_code }" )

        passed_companies = []
        for comp in filtered_companies :
            address_data = comp.get( 'address' , '' )
            address_str = str( address_data ).lower()
            comp_country_code = ""

            if isinstance( address_data , dict ) :
                comp_country_code = str( address_data.get( 'country_code' , '' ) ).lower()
            elif isinstance( address_data , str ) :
                match = re.search( r"['\"]country_code['\"]\s*:\s*['\"]([a-z]{2})['\"]" , address_str )
                if match :
                    comp_country_code = match.group( 1 )

            if comp_country_code == target_country_code or target_country_name in address_str :
                passed_companies.append( comp )

        filtered_companies = passed_companies

    print( f"[ Filter 1 ] result : Passed { len( filtered_companies ) } companies from { len( companies ) }" )
    return filtered_companies




# Third layer of filtering ( it was supposed to be the second layer ,but I changed the order of the filters ) :)
# Business Model Clustering ( Jaccard Score + HAC )

def extract_business_words( comp ) :

    """
    Extracts a relevant bag of words ( Set ) to describe what the company does .
    This includes : NAICS , Core Offerings , Target Markets , Description
    """

    words = []

    # Primary_naics
    naics = comp.get( 'primary_naics' , '' )
    if isinstance( naics , dict ) :
        words.extend( str( naics.get( 'label' , '' ) ).lower().split() )
    elif isinstance( naics , str ) :
        match = re.search( r"['\"]label['\"]\s*:\s*['\"]([^'\"]+)['\"]" , str( naics ).lower() )
        if match : words.extend( match.group( 1 ).split() )
    # Core offerings & Target Markets
    for key in [ 'core_offerings' , 'business_model' ] :
        items = comp.get( key , [] )
        if isinstance( items , list ) :
            for item in items :
                item_str = str( item ).lower()
                words.extend( item_str.split() )

                # Lexical Trick
                if 'business-to-bussiness' in item_str : words.append( 'b2b' )
                if 'business-to-consumer' in item_str : words.append( 'b2c' )
                if 'software-as-a-service' in item_str : words.append( 'saas' )

    # Description
    desc = comp.get( 'description' , '' )
    if isinstance( desc , str ) :
        words.extend( desc.lower().split() )

    # eliminating the stop words
    clean_words = set( w for w in words if len( w ) > 3 and w not in STOP_WORDS_QUERY )
    for tag in [ 'b2b' , 'b2c' , 'saas' ] :
        if tag in words : clean_words.add( tag )

    return clean_words

def jaccard_distance( set1 , set2 ) :

    """
    Calculates the Jaccard distance between two sets.
    0.0 means both sets are identical
    1.0 means both sets are completely different
    """

    if not set1 and not set2 : return 0.0
    intersection = len( set1.intersection( set2 ) )
    union = len( set1.union( set2 ) )
    return 1.0 - ( intersection / union )

def business_model_clustering( query , companies ) :

    """
    Second filter : this method groups companies taking in consideration their business model
    The algorithm groups them using Hierarchical Clustering and keeps the cluster with the highest BM25 score
    """

    if len( companies ) <= 1 :
        return companies
    print( f"\n[ Filter 3 ] starting clustering for { len( companies ) } companies ..." )

    lexical_profiles = [ extract_business_words( c ) for c in companies ]
    n = len( companies )
    dist_matrix = np.zeros( ( n, n ) )

    for i in range( n ) :
        for j in range( n ) :
            dist_matrix[ i ][ j ] = jaccard_distance( lexical_profiles[ i ] , lexical_profiles[ j ] )

    clusterer = AgglomerativeClustering(
        n_clusters = None ,
        distance_threshold = 0.88 ,
        metric = 'precomputed' ,
        linkage = 'average'
    )
    labels = clusterer.fit_predict( dist_matrix )

    # Eliminating the negations
    negative_words = set()
    negation_patterns = r'(?:not|exclude|excluding|without|no|except)\s+(?:to\s+be\s+involved\s+in\s+|involved\s+in\s+)?([a-z\s]+)'
    matches = re.finditer( negation_patterns , query.lower() )
    for match in matches :
        words_after = match.group( 1 ).split()
        for w in words_after[:2] :
            if len( w ) > 2 and w not in STOP_WORDS_QUERY :
                negative_words.add( w )

    # Final step is to identify the best cluster
    query_words = set( w for w in query.lower().split() if len( w ) > 2 and w not in STOP_WORDS_QUERY )
    query_words = query_words - negative_words
    cluster_scores = {}

    for i , cluster_id in enumerate( labels ) :
        if cluster_id not in cluster_scores :
            cluster_scores[ cluster_id ] = 0

        overlap = len( lexical_profiles[ i ].intersection( query_words ) )
        bm25_score = companies[ i ].get( 'bm25_score' , 0 )
        cluster_scores[ cluster_id ] += ( overlap * 15 ) + ( bm25_score * 0.1 )

    best_cluster_id = max( cluster_scores , key = cluster_scores.get )
    print( f"[ Filter 3 ] created { len( set( labels ) ) } clusters . Winning cluster : ID = { best_cluster_id } , Score = { cluster_scores[ best_cluster_id ]:.2f}" )

    passed_companies = [ comp for i , comp in enumerate( companies ) if labels[ i ] == best_cluster_id ]
    print( f"[ Filter 3 ] result : Passed { len( passed_companies ) } companies from the winning business cluster ..." )
    return passed_companies




# Second layer of filtering ( Intent & Negation )
# The finisher of Stage 0

def intent_and_negation_filter( query , companies ) :

    if not companies :
        return []

    print( f"\n[ Filter 2 ] checking negations and business intent ... " )
    query_lower = query.lower()
    negative_words = set()
    negation_patterns = r'(?:not|exclude|excluding|without|no|except)\s+(?:to\s+be\s+involved\s+in\s+|involved\s+in\s+)?([a-z\s]+)'
    matches = re.finditer( negation_patterns , query_lower )

    for match in matches :
        words_after = match.group( 1 ).split()
        for w in words_after[:2] :
            if len( w ) > 2 and w not in STOP_WORDS_QUERY :
                negative_words.add( w )

    if negative_words :
        print( f"[ Filter 2 ] Negations detected : { negative_words } " )

    req_b2b = 'b2b' in query_lower or 'business to business' in query_lower
    req_b2c = 'b2c' in query_lower or 'business to consumer' in query_lower
    req_saas = 'saas' in query_lower or 'software as a service' in query_lower

    passed_companies = []

    for comp in companies :

        profile_words = extract_business_words( comp )
        if negative_words and len( profile_words.intersection( negative_words ) ) > 0 :
            continue

        biz_models = [ str( m ).lower() for m in comp.get( 'business_model' , [] ) ]
        biz_model_text = " ".join( biz_models )

        if req_b2b and 'business-to-consumer' in biz_model_text and 'business-to-business' not in biz_model_text :
            continue
        if req_b2c and 'business-to-business' in biz_model_text and 'business-to-consumer' not in biz_model_text :
            continue
        if req_saas and 'software' not in profile_words and 'saas' not in biz_model_text :
            continue

        passed_companies.append( comp )

    print( f"[ Filter 2 ] result : Passed { len( passed_companies ) } companies ." )
    return passed_companies










def run_cascade( query , bm25_results_list ) :
    print( "\n[ Cascade Pipeline ] Starting filtering and clustering ... " )

    step_1_results = geo_temporal_employee_filter( query , bm25_results_list )
    step_2_results = intent_and_negation_filter( query , step_1_results )
    step_3_results = business_model_clustering( query , step_2_results )

    return step_3_results

