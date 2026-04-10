import pycountry

STOP_WORDS_QUERY = {
    # prepositions and pronouns
    "the" , "and" , "for" , "with" , "that" , "this" , "are" , "from" , "their" , "our" , "its" , "has" , "have" , "who" , "which" , "where" , "what" , "in" , "on" , "at" , "to" , "of" , "about" ,
    # search intent
    "find" , "looking" , "search" , "show" , "give" , "top" , "best" , "list" , "all" , "specialized" , "specializing" , "companies" , "company" , "startup" , "startups" , "enterprise" , "business" , "firm" , "firms" , "agency" , "provider" , "providers" , "services" , "industry" , "sector" ,
    # constraints
    "more" , "than" , "before" , "after" , "over" , "minimum" , "at" , "least" , "employees" , "employee" , "people" , "staff" , "revenue" , "year" , "years" , "founded" , "established" , "located" , "based" ,
    # negations
    "not" , "no" , "exclude" , "excluding" , "without" , "other" , "except"
}

def build_countries_map() :
    """
    Building a dictionary that maps country codes to a list of all countries
    """

    cmap = {}
    for country in pycountry.countries :
        alpha_2 = country.alpha_2.lower()
        main_name = country.name.lower()
        keywords = [ main_name ]

        if hasattr( country , 'official_name' ) :
            keywords.append( country.official_name.lower() )

        # Add some well known exceptions
        if alpha_2 == 'de': keywords.extend( [ 'deutschland' ] )
        if alpha_2 == 'ro': keywords.extend( [ 'romania' ] )
        if alpha_2 == 'fr': keywords.extend( [ 'franta' , 'france' ] )
        if alpha_2 == 'es': keywords.extend( [ 'españa' , 'spain' ] )
        if alpha_2 == 'ch': keywords.extend( [ 'schweiz' ] )
        if alpha_2 == 'gb': keywords.extend( [ 'great britain' , 'uk' , 'united kingdom' ] )
        if alpha_2 == 'us': keywords.extend( [ 'usa' , 'america' , 'sua' , 'united states of america' ] )

        cmap[ main_name ] = ( alpha_2 , keywords )

    return cmap

COUNTRIES_MAP = build_countries_map()

for c_name in COUNTRIES_MAP.keys() :
    STOP_WORDS_QUERY.add( c_name )
    for kw in COUNTRIES_MAP[ c_name ][ 1 ] :
        STOP_WORDS_QUERY.add( kw )