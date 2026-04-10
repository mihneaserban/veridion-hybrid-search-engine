from flask import Flask , request , jsonify , render_template
import webbrowser
import threading
import logging
import os
import json
import traceback

log = logging.getLogger( 'werkzeug' )
log.setLevel( logging.ERROR )

class SearchInterface :

    def __init__( self , search_callback ) :

        self.search_callback = search_callback
        template_dir = os.path.abspath( os.path.join( os.path.dirname( __file__ ) , 'templates' ) )
        self.app = Flask( __name__ , template_folder = template_dir )
        self.setup_routes()

    def setup_routes( self ) :

        @self.app.route('/')
        def home() :
            return render_template( 'index.html' )

        @self.app.route( '/api/search' , methods=[ 'POST' ] )
        def search() :
            try :
                data = request.json
                query = data.get( 'query' )
                print( f"\n[ Web UI ] Getting query : { query }" )

                output_path , results = self.search_callback( query )
                clean_results_for_web = []
                for comp in results :
                    clean_results_for_web.append( {
                        "operational_name": comp.get( 'operational_name' , "Unknown" ) ,
                        "description": comp.get( 'description' , "No description available" ) ,
                        "hybrid_final_score": float( comp.get( 'hybrid_final_score' , 0.0 ) )
                    } )

                return jsonify( {
                    'output_path' : output_path,
                    'results' : clean_results_for_web
                } )
            except Exception as e :
                error_msg = str( e )
                print( f"\n[ Python Error ] :\n")
                traceback.print_exc()
                return jsonify( { 'error' : error_msg } ) , 500

    def run( self , port = 5000 ) :

        print( f"\n[ Veridion UI ] Starting server ... Opening in your browser : http://127.0.0.1:{port}" )
        threading.Timer( 1.25 , lambda: webbrowser.open(f'http://127.0.0.1:{port}') ).start()
        self.app.run( port = port , debug = False , use_reloader = False )
