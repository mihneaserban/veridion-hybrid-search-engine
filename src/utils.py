import pandas as pd
import os
import re
import math
import json

def load_data( file_path ) :

    if not os.path.exists( file_path ) :
        raise FileNotFoundError( f"Error: File { file_path } not found!" )

    return pd.read_json( file_path , lines = True )

def preprocess_df( df ) :

    for col in df.columns :

        def clean_value( val ) :
            if isinstance( val , list ) :
                val_str =  " ".join( str( item ) for item in val )
            elif val is None or val == '' :
                return ""
            elif isinstance( val , float ) and pd.isna( val ) :
                return ""
            elif isinstance( val , float ) :
                if val.is_integer() :
                    val_str = str( int( val ) )
                else :
                    val_str = str( val )
            else :
                val_str = str( val )
            clean_str = re.sub( r"[{}[\]\"',]" , " " , val_str )

            return " ".join( clean_str.split() )

        df[ col ] = df[ col ].apply( clean_value )

    return df


def impute_missing_financials( input_path , output_path ) :

    """
        Going through JSONL file and fix fields 'employee_count' and 'revenue' which are empty ( NaN ).
        This function also saves the result in a new file because I do not want to alter the original file .
    """

    if not os.path.exists( input_path ) :
        raise FileNotFoundError( f"Error: File { input_path } not found!" )

    REVENUE_PER_EMPLOYEE = 130000.0
    DEFAULT_EMPLOYEES = 10.0

    def is_valid_number( val ) :
        if val is None or val == '' :
            return False
        try :
            f_val = float( val )
            return not math.isnan( f_val )
        except ( ValueError, TypeError ) :
            return False

    processed_count = 0
    fixed_count = 0

    with open( input_path , 'r' , encoding = 'utf-8' ) as infile , \
        open( output_path, 'w' , encoding = 'utf-8' ) as outfile :

        for line in infile :
            if not line.strip() : continue

            comp = json.loads( line )
            emp = comp.get( 'employee_count' )
            rev = comp.get( 'revenue' )
            has_emp = is_valid_number( emp )
            has_rev = is_valid_number( rev )


            # Case 1 : we have employees
            if has_emp and not has_rev :
                comp[ 'revenue' ] = float( emp ) * REVENUE_PER_EMPLOYEE
                # mark
                comp[ 'revenue_note' ] = "Estimated based on employee count ( ~ 130k/emp )"
                fixed_count += 1
            # Case 2 : we have revenue
            elif has_rev and not has_emp :
                comp[ 'employee_count' ] = round( float( rev ) / REVENUE_PER_EMPLOYEE )
                # mark
                comp[ 'employee_count_note' ] = "Estimated based on revenue ( ~ 130k/emp )"
                fixed_count += 1
            # Case 3 : we have nothing
            elif not has_emp and not has_rev :
                comp[ 'employee_count' ] = DEFAULT_EMPLOYEES
                comp[ 'revenue' ] = DEFAULT_EMPLOYEES * REVENUE_PER_EMPLOYEE
                comp[ 'employee_count_note' ] = "Estimated ( default SME )"
                comp[ 'revenue_note' ] = "Estimated ( default SME )"
                fixed_count += 1

            json.dump( comp , outfile , ensure_ascii = False )
            outfile.write( '\n' )
            processed_count += 1

    print( f"[ Utils ] Processed { processed_count } companies ... " )
    print( f"[ Utils ] Fixed { fixed_count } companies ... " )
    print( f"[ Utils ] The clean file was saved in '{ output_path }' " )
