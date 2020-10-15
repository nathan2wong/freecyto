import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, make_response, session, abort
from werkzeug.utils import secure_filename
#import data_visualize
import selective_data_vis
import ast
import datetime
import random
import hashlib
from base64 import b64encode
import sqlite3
import pandas as pd
import time

import advanced_data_vis

#Google Firebase settings
from google.auth.transport import requests
from google.cloud import datastore
import google.oauth2.id_token
firebase_request_adapter = requests.Request()
datastore_client = datastore.Client()

#Flask settings
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data')
ALLOWED_EXTENSIONS = set(['fcs'])
app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Checks the name of the file uploaded to see if it contains a valid extension

    Parameters
    ----------
    filename : str
        Name of the file uploaded.
    Returns
    -------
    boolean
        True if the file has an approved extension (in ALLOWED_EXTENSIONS)

    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_exp(email, limit):
    """
    Queries all entries stored under the email (under specified limit).

    Data stored is under the collection "quickdata" in Google Firestore.

    Parameters
    ----------
    email : str
        Email of user to query results from.
    limit : int
        Maximum number of results to be queried
    Returns
    -------
    list
        Past experiments performed by the user with the inputted email

    """
    ancestor = datastore_client.key('User', email)
    query = datastore_client.query(kind="quickdata", ancestor=ancestor)
    #query.order = ['-timestamp']
    output = query.fetch(limit=limit)
    return output

def store_url(email, args, kind):
    """
    Stores the URL of a quick visualization data visualization procedure, so the user can call the same experiment
    easily.

    Removes lengthy URLs from the stored arguments ('html_graph', 'form_html'), and adds a timestamp.

    Parameters
    ----------
    email : str
        Email of the user to store data
    args : dict
        dict_keys(['form_html', 'available_cols', 'csv', 'images_to_display', 'filenames', 'boxwhisker_stats',
        'describe', 'selected', 'html_graph', 'transformation', 'exp_files', 'directory', 'used_cols', 'experimental',
        'int_hash'])
        URL parameters to store
    kind : str
        Name of collection to store the Entity for Firestore

    """
    copy_args = dict(args)
    copy_args.pop('html_graph')
    copy_args.pop('form_html')
    entity = datastore.Entity(key=datastore_client.key('User', email, kind))
    copy_args['timestamp'] = datetime.datetime.now()
    entity.update(copy_args)
    datastore_client.put(entity)

def checkLoggedIn():
    """
    Checks to see if a user is logged in.

    if ANY user is logged in – data is shared between all authenticated users. This is performed based on the
    Firebase cookie stored when an account is created / logged into.

    Returns
    -------
    boolean
        True if the id_token can be verified by firebase, and thus represents a user that is properly authenticated

    """
    id_token = request.cookies.get("token")
    if id_token:
        try:
            claims = google.oauth2.id_token.verify_firebase_token(
                id_token, firebase_request_adapter)
            #store_url(claims['email'], {"date": datetime.datetime.now()}, 'visit')
            return True
        except ValueError as exc:
            # This will be raised if the token is expired or any other
            # verification checks fail.
            print(exc)
    return False

def exists(email, args):
    """
    Checks to see if an experiment has already been performed. Need to optimize the query parameters.

    Parameters
    ----------
    email : str
        Name of the file uploaded.
    args : dict
        URL parameters to check
        {'exp_files': ['filename.fcs'], 'directory': '2019-06-11 ...',
        'transformation': 'raw', 'selected': [3, 5]}
    Returns
    -------
    dict
        Full URL parameters if the user-inputted arguments has a match with the entries in datastore records.
        Otherwise, return None.

    """
    ancestor = datastore_client.key('User', email)
    query = datastore_client.query(kind="quickdata", ancestor=ancestor)
    #query.order = ['-timestamp']
    #query.add_filter("transformation", "=", "raw")
    output = list(query.fetch(limit=50))

    for o in output:
        if all([o[key] == args[key] for key in args.keys() if key != "directory" and key != "int_hash" and key != "tSNE_O" and key != "KDE"]):
            return o
    return None

@app.route('/')
def home():
    """
    Flask route for homepage.

    Returns
    -------
    Render for home.html
       Flask render for homepage
    Function to upload file if logged in *will not happen since firebase is being used instead of sqlalchemy
    to store user account information.

    """
    if not session.get('logged_in'):
        login = url_for('do_admin_login')
        return render_template('home.html', login=login)
    else:
        return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Flask route to upload a file.

    First checks to see if the user is logged in, otherwise force the user to log in.
    Next, if the user has submitted an uploaded file (via POST request), then proceed to call_functions(),
    in which visualization protocols are performed.
    Otherwise, render the upload.html template

    Returns
    -------
    call_functions(args=args)
        Route to visualization protocols if appropriate file(s) were uploaded
    render upload.html
        Display upload page if appropriate file(s) have not been uploaded yet

    """
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))
    if request.method == 'POST':
        exp_files = []
        date = str(datetime.datetime.now())
        CUR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], date)
        os.mkdir(CUR_DIR)
        transformation = request.form['transformation']
        try:
            tSNE_O = request.form['tSNE_O']
        except:
            tSNE_O = None
        try:
            KDE = request.form["KDE"]
        except:
            KDE = None

        for file in request.files.getlist("expFiles"):
            print(file)
            if file.filename == '' or not allowed_file(file.filename):
                return redirect(request.url)
            else:
                exp = secure_filename(file.filename)
                file.save(os.path.join(CUR_DIR, exp))
                exp_files.append(exp)

        #Defaulted Selected Channels
        selected = [3,5]

        args = {"exp_files": exp_files, "directory": date, "transformation": transformation,
                "selected": selected, "tSNE_O": tSNE_O, "KDE": KDE}
        return redirect(url_for('call_functions', args=args, tSNE_O=tSNE_O, KDE=KDE))
    return render_template("upload.html", upload=True)

#For testing
@app.route("/<path:path>")
def images(path):
    resp = make_response(open(path).read())
    resp.content_type = "image/png"
    return resp

@app.route('/visualization')
def uploaded_file():
    """
    Flask route when trying to retrieve a file.

    First obtains the current directory and the filename from the GET argument parameters,
    and then calls Flask's native send_from_directory method to obtain an URI for the
    file at that location.

    Returns
    -------
    URI
        for the file and its fullpath location

    """
    CUR_DIR = request.args.get('directory')
    filename = request.args.get('filename')
    return send_from_directory(CUR_DIR,
                               filename)

@app.route('/run')
def call_functions():
    """
    Flask route to run the quick visualization options.

    args = {"exp_files": exp_files, "directory": date, "transformation": transformation,
        "selected": selected}
    1. Checks to see if the experiment with the above args parameters has already been performed by this uses / email
    address.
        If it has, get the arguments and reformulate the URL for html_graph and form_html and render quick_vis.html
        using the stored arguments.
        Otherwise:
    2. Create the experiment analysis using the data_vis package in this project, and obtain a list of all images and their
    respective contents to display.
    3. Store the new experiment, and render the quick_vis.html with the newly created arguments.

    Returns
    -------
    render quick_vis.html
        args = {"form_html": form_html, "available_cols": available_cols, "csv": csv, "images_to_display": images_to_display,
            "filenames": filenames, "boxwhisker_stats": boxwhisker_stats, "describe":describe, "selected": selected, "html_graph": html_graph, "transformation": transformation,
            "exp_files": args['exp_files'], "directory": args['directory'], "used_cols": used_cols, "experimental": experimental, "int_hash": int_hash}

    """
    print("Start time:", time.time())
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))

    args = ast.literal_eval(request.args.get('args'))
    CUR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], args['directory'])

    #authenticate and check if already done
    id_token = request.cookies.get("token")
    claims = google.oauth2.id_token.verify_firebase_token(
        id_token, firebase_request_adapter)
    email = claims['email']
    res = exists(email, args)
    print(res)
    if res is not None:
        args['available_cols'] = res['available_cols']
        args['int_hash'] = res['int_hash']
        res['html_graph'] = url_for('html_graph', args=args)
        res['form_html'] = url_for('change', args=args)
        return render_template("quick_vis.html", args=res)

    #If jobs has not yet been done
    #random_bytes = urandom(64)
    #token = b64encode(random_bytes).decode('utf-8')
    int_hash = random.randint(1, 65536) + random.randint(1, 1337)
    args['int_hash'] = int_hash
    transformation = args['transformation']
    selected = [int(i) for i in args['selected']]

    directory = os.path.join(CUR_DIR, str(int_hash))
    if not os.path.exists(directory):
        os.mkdir(directory)
    try:
        tSNE_O = args["tSNE_O"]
    except:
        tSNE_O = False
    try:
        KDE = args["KDE"]
    except:
        KDE = False

    exp_wrapper = selective_data_vis.constructExperiment(directory, args['exp_files'], transformation, CUR_DIR, tSNE_O, KDE)
    selective_data_vis.performDataAnalyses(exp_wrapper, selected, args['exp_files'])

    images_to_display = []
    filenames = []
    boxwhisker_stats = []
    describe = []
    experimental = {}

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            if 'distribution overview' in filename or 'Combined Boxplot' in filename:
                experimental[filename] = url_for('uploaded_file', filename=filename, directory=directory)
            else:
                if 'KDE' in filename:
                    images_to_display.insert(0, url_for('uploaded_file', filename=filename, directory=directory))
                    filenames.insert(0, filename)
                else:
                    images_to_display.append(url_for('uploaded_file', filename=filename, directory=directory))
                    filenames.append(filename)
        elif filename == "stats.txt":
            file_stats = open(os.path.join(directory, filename), "r")
            for line in file_stats.readlines():
                all_stats = line.split("{")
                for stat in all_stats:
                    end = stat.split("}")
                    if len(end) > 1:
                        boxwhisker_stats.append(end[0])
        elif filename == "describe.txt":
            file_desc = open(os.path.join(directory, filename), "r")
            file_split = file_desc.read().split("\n")

            collect = False
            col_desc = []

            for part in file_split:
                start = part.split("{")
                end = part.split("}")

                if(len(end) > 1):
                    describe.append(str(col_desc))
                elif(collect):
                    col_desc.append(part)
                if(len(start) > 1):
                    collect = True
                    col_desc = []
                    col_desc.append(start[1])

    csv = url_for('uploaded_file', filename= "alldata.xlsx", directory=directory)

    available_cols = []
    colnames = exp_wrapper.columns
    for index in range(len(colnames)):
        available_cols.append(colnames[index])

    args['available_cols'] = available_cols
    form_html=url_for('change', args=args)
    html_graph = url_for('html_graph', args=args)
    used_cols = [available_cols[int(i)] for i in selected]
    args = {"form_html": form_html, "available_cols": available_cols, "csv": csv, "images_to_display": images_to_display,
            "filenames": filenames, "boxwhisker_stats": boxwhisker_stats, "describe":describe, "selected": selected, "html_graph": html_graph, "transformation": transformation,
            "exp_files": args['exp_files'], "directory": args['directory'], "used_cols": used_cols, "experimental": experimental, "int_hash": int_hash}
    store_url(email, args, 'quickdata')
    print("Quick time:", time.time())
    return render_template("quick_vis.html", args=args)

@app.route('/login', methods=['GET','POST'])
def do_admin_login():
    """
    Flask route to perform an authentication check for the current user using the cookies "token".

    Simply check is any user is authenticated, by connecting the Firebase auth system. Otherwise, return to the
    user login page. Also, if the user is authenticated, it will list the top 50 most recent jobs.

    Note: This method is temporary – fetching experiments every time authentication is required is not efficient.

    Returns
    -------
    render user.html
        returns to home page, and with a list of all recently performed experiments.

    """
    id_token = request.cookies.get("token")
    exps = None

    if id_token:
        try:
            claims = google.oauth2.id_token.verify_firebase_token(
                id_token, firebase_request_adapter)
            #store_url(claims['email'], {"date": datetime.datetime.now()}, 'visit')
            exps = fetch_exp(claims['email'], 50)

        except ValueError as exc:
            # This will be raised if the token is expired or any other
            # verification checks fail.
            error_message = str(exc)
            print(error_message)
    return render_template('user.html', exps=exps)

@app.route('/change_vis', methods=['GET', 'POST'])
def change():
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))

    """
    args =
    {'exp_files': ['4b.fcs', '6a.fcs'], 'directory': '2019-06-06 13:06:11.820318', 'transformation': 'raw',
    'selected': [3, 5], 'available_cols': ['Forward Scatter (FSC-HLin)', '...']}
    """
    args = ast.literal_eval(request.args.get('args'))
    if request.method == 'POST':
        # POST_COL1 and COL2 are the names of the columns that will be used
        POST_COLS = request.form.getlist('channels')
        args['selected'] = POST_COLS
        return redirect(url_for('call_functions', args=args))

    return redirect(request.url)

##Adv
@app.route('/html_graph')
def html_graph():
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))
    print("Start adv time:", time.time())
    args = ast.literal_eval(request.args.get('args'))

    available_cols = args['available_cols']
    selected = [int(s) for s in args['selected']]

    CUR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], args['directory'])
    int_hash = args['int_hash']
    directory = os.path.join(CUR_DIR, str(int_hash))
    files = args['exp_files']

    sqlexp = advanced_data_vis.initiateAnalysis(directory=directory, selected_cols=selected, available_cols=available_cols, files=files)

    args['option'] = 0
    args['num_clusters'] = sqlexp.num_clusters
    return redirect(url_for('advanced_analysis', args=args))


@app.route('/advanced_analysis', methods=['GET', 'POST'])
def advanced_analysis():
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))

    args = ast.literal_eval(request.args.get('args'))
    available_cols = args['available_cols']
    selected = args['selected']
    counter = args['int_hash']
    transformation = args['transformation']
    num_clusters = args['num_clusters']
    def_xlim, def_ylim = 0,0

    render = dict(args)

    CUR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], args['directory'])
    files = args['exp_files']

    selected = [int(s) for s in selected]
    directory = os.path.join(CUR_DIR, str(args['int_hash']))
    option = int(args['option'])
    cachebust = str(random.getrandbits(50))
    reset = url_for('reset_sql', args=args)
    sql = url_for('uploaded_file', filename= "experiment.db", directory=directory)

    render['reset'] = reset
    render['cacheburst'] = cachebust
    #directory, selected_cols, available_cols, files, option, selected=[0,1], xlimit=-1, ylimit=-1, cachebust="",
    #redir_url=None, noChange=True

    urls = []
    titles = []
    for num in range(len(files)+3):
        args_copy = dict(args)
        args_copy['option'] = num
        url = url_for('advanced_analysis', args=args_copy)
        urls.append(url)
        if num < len(files):
            titles.append(files[num])
        elif num == len(files):
            titles.append("Combined Analysis")
        elif num == len(files) + 1:
            titles.append("Coordinate Gated")
        elif num == len(files) + 2:
            titles.append("Deep Gating")
        else:
            titles.append(str(num))

    render['urls'] = urls
    render['titles'] = titles

    gating_cols = [0,1]
    if request.method == 'POST':
        gating_cols = request.form.getlist('x_channels') + request.form.getlist('y_channels')
        gating_cols = [int(c) for c in gating_cols]

        if option == len(files) + 1:
            def_xlim = float(request.form.get('xlimit'))
            def_ylim = float(request.form.get('ylimit'))
            advanced_data_vis.generateHTML(directory=directory, selected_cols=selected, available_cols=available_cols, files=files,
                                       option=option, selected=gating_cols, xlimit=def_xlim, ylimit=def_ylim, cachebust=cachebust)
    else:
        def_xlim, def_ylim = advanced_data_vis.generateHTML(directory=directory, selected_cols=selected, available_cols=available_cols, files=files,
                               option=option, selected=gating_cols, xlimit=0, ylimit=0, cachebust=cachebust)

    render['gating_cols'] = gating_cols
    render['xlim'] = def_xlim
    render['ylim'] = def_ylim

    #single file graphing
    if option in range(len(files)):
        singleHTML = url_for('uploaded_file', filename= "{0}kmeans_{1}gating.html".format(cachebust, files[option]), directory=directory)
        print("End Adv time:", time.time())
        return render_template("dashboard.html", HTMLGraph=singleHTML, myself=files[option],
                               urls=urls, selected=selected, cachebust=cachebust, reset=reset,
                               transformation=transformation, files=files, titles=titles, render=render, num_clusters=num_clusters, sql=sql)
    #Combined graphing
    if option == len(files):
        combined = url_for('uploaded_file', filename="{0}combined_gating.html".format(cachebust), directory=directory)
        return render_template("dashboard.html", HTMLGraph=combined, myself="Combined Analysis",
                               urls=urls, selected=selected, cachebust=cachebust, reset=reset,
                               transformation=transformation, files=files, titles=titles, render=render, num_clusters=num_clusters, sql=sql)
    #Coordinate gating
    if option == len(files) + 1:
        form_html = url_for('advanced_analysis', args=args)
        coordinates = []
        for file in files:
            coordinate_gating = url_for('uploaded_file', filename="coordinate_gating_kmeans_{0}{1}.html".format(file, cachebust), directory=directory)
            coordinates.append(coordinate_gating)
        return render_template("dashboard.html", graphs=coordinates,
                               myself="Coordinate Gated Analysis", urls=urls, form_html=form_html, gating_cols=gating_cols,
                               selected=selected, available_cols=available_cols, def_xlim=def_xlim, def_ylim=def_ylim, cachebust=cachebust, reset=reset,
                               transformation=transformation, files=files, titles=titles, render=render, num_clusters=num_clusters, sql=sql)
    #Deep gate
    if option == len(files) + 2:
        args['cacheburst'] = cachebust
        form_html_channel = url_for('advanced_analysis', args=args)
        redir_url = {}

        for file in files:
            redir_url[file] = url_for('get_data', args=args, file=file)
        if request.method == "GET":
            changechannel = True
        else:
            changechannel = False
        advanced_data_vis.generateHTML(directory=directory, selected_cols=selected, available_cols=available_cols, files=files,
                                       option=option, selected=gating_cols, xlimit=def_xlim, ylimit=def_ylim, cachebust=cachebust,
                                       redir_url=redir_url, noChange=changechannel)
        full_gates = []
        for file in files:
            full_gate = url_for('uploaded_file', filename="{0}gating{1}.html".format(file, cachebust), directory=directory)
            full_gates.append(full_gate)
        return render_template("dashboard.html", graphs=full_gates, myself="Deep Gating Analysis",
                               form_html_channel=form_html_channel, gating_cols=gating_cols, available_cols=available_cols,
                               urls=urls, selected=selected, reset=reset, transformation=transformation, files=files, titles=titles, render=render, num_clusters=num_clusters, sql=sql)

@app.route('/gating', methods=['GET', 'POST'])
#Need fix
#Idea is to simply check data that is posted (indices) against the "index" values of the actual sqldb
#If "index" resets to 0 everytime, then add a line in parseSQL that negates lookup of the new 'indices' column of actual sqldb
def get_data():
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))

    args = ast.literal_eval(request.args.get('args'))

    available_cols = args['available_cols']
    selected = args['selected']
    CUR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], args['directory'])
    files = args['exp_files']
    selected = [int(s) for s in selected]
    directory = os.path.join(CUR_DIR, str(args['int_hash']))
    cachebust = args['cacheburst']
    file_to_change = request.args.get("file")

    if request.method == 'POST':
        data = request.form['json']
        indices = ast.literal_eval(data)
        indices = [int(i)+1 for i in indices]
        pd_indices = pd.DataFrame(indices, columns=["indices"])

        db_file = os.path.join(directory, "experiment.db")
        conn = sqlite3.connect(db_file)

        indices_table = "indices" + str(cachebust)
        pd_indices.to_sql(indices_table, conn)

        def changeSQL(file, tablename):
            conn.execute("ALTER TABLE 'kmeans_{0}' RENAME TO '{1}'".format(file, tablename))
            controlstatement = """CREATE TABLE 'kmeans_{0}' AS SELECT * FROM '{1}' WHERE rowid in
                           (SELECT indices FROM {2});""".format(file, tablename, indices_table)
            conn.execute(controlstatement)
            conn.execute("""CREATE TABLE 'og_kmeans_{0}' AS SELECT * FROM '{1}'""".format(file, tablename))

        og_kmean = "kmeans_{0}{1}".format(file_to_change, cachebust)
        changeSQL(file_to_change, og_kmean)

        conn.close()
    return "Success"

@app.route('/reset')
def reset_sql():
    if not checkLoggedIn():
        return redirect(url_for('do_admin_login'))

    args = ast.literal_eval(request.args.get('args'))
    counter = args['int_hash']
    files = args['exp_files']

    CUR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], args['directory'])
    directory = os.path.join(CUR_DIR, str(args['int_hash']))

    db_file = os.path.join(directory, "experiment.db")
    conn = sqlite3.connect(db_file)

    for file in files:
        try:
            if len(conn.execute("SELECT * FROM 'og_kmeans_{0}'".format(file)).fetchall()) > 0:
                conn.execute("DROP TABLE 'kmeans_{0}'".format(file))
                conn.execute("ALTER TABLE 'og_kmeans_{0}' RENAME TO 'kmeans_{0}'".format(file))
        except:
            pass
    time.sleep(3)
    return "Success"
