import pandas as pd
import holoviews as hv
from holoviews import dim, opts
from holoviews.operation import histogram
hv.extension('bokeh')
import sqlite3
from sklearn.cluster import KMeans
import numpy as np
import os
from multiprocessing import Process
from multiprocessing import Manager


class SQLVis:
    def __init__(self, directory, selected_cols, columns, files):
        self.db_file = os.path.join(directory, "experiment.db")
        self.directory = directory
        self.all_cols = columns
        self.columns = []
        self.selected = list(selected_cols)
        self.files = files
        self.exps = {}
        opts.defaults(opts.Points(fontsize={'title': 18, 'labels': 18}))

    def sqldb_kmeans(self):
        conn = sqlite3.connect(self.db_file)
        for file in self.files:
            try:
                self.decimated[file].to_sql("kmeans_{0}".format(file), conn)
            except:
                pass
        conn.close()

    def parseSQL(self, special_files):
        conn = sqlite3.connect(self.db_file)
        c = conn.execute("PRAGMA table_info('{0}')".format(special_files[0]))
        output = c.fetchall()

        for cols in output:
            if 'index' in cols[1]:
                continue
            self.columns.append(cols[1])
        converted_cols = []
        for index in self.selected:
            converted_cols.append(self.all_cols[index])

        converted_cols.extend(list(self.findColSSCFSC()))
        converted_cols = list(set(converted_cols))
        ret = {}
        for file in special_files:
            statement = "SELECT {0} FROM '{1}';".format(','.join('"{0}"'.format(c) for c in converted_cols), file)
            data = pd.read_sql(sql=statement,con=conn)
            ret[file] = data
            self.exps[file] = data
        conn.close()
        return ret
    def findColSSCFSC(self):
        SSC = ""
        FSC = ""
        for col in self.columns:
            if 'SSC' in col:
                if 'A' in col or len(SSC) == 0:
                    SSC = col
            if 'FSC' in col:
                if 'A' in col or len(FSC) == 0:
                    FSC = col
        return SSC, FSC
    def decimateKMeans(self, n_clusters=250):
        manager = Manager()
        return_dict = manager.dict()
        def worker(data, clusters, file):
            km = KMeans(n_clusters=clusters, n_init=5, random_state=0).fit(data)
            clusters = km.cluster_centers_
            clustered = pd.DataFrame(clusters)
            clustered.columns = data.columns
            return_dict[file] = clustered

        jobs = []
        for file in self.files:
            p = Process(target=worker, args=(self.exps[file], n_clusters, file))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        self.decimated = return_dict
        return self.decimated

    def generateHoloview(self, df, SSC, FSC, type, counter_max=50):
        renderer = hv.renderer('bokeh')
        body_points = hv.Scatter(df, SSC, FSC).opts(color='r', title='SSC vs FSC Default Gating')
        body_hist = body_points.hist(num_bins=50, dimension=[SSC, FSC])
        body = body_hist

        counter = 0

        for index in range(len(df.columns)):
            for index2 in range(index+1, len(df.columns)):
                col = df.columns[index]
                col2 = df.columns[index2]
                if col2 != col and col not in (SSC, FSC) and col2 not in (SSC, FSC) and counter < counter_max:
                    points = hv.Scatter(df, col, col2)
                    hist = points.hist(num_bins=50, dimension=[col, col2])
                    body += hist
                    counter += 1
        print(counter)
        try:
            body = body.opts(
                opts.Scatter(tools=['box_select', 'lasso_select']),
                opts.Layout(shared_axes=True, shared_datasource=True)).cols(2)
        except:
            body = body.opts(
                opts.Scatter(tools=['box_select', 'lasso_select']),
                opts.Layout(shared_axes=True, shared_datasource=True))
        renderer.save(body, os.path.join(self.directory, str(type)+"gating"))

    def generateCombined(self, decimated, SSC, FSC, cachebust, counter_max=20):
        renderer = hv.renderer('bokeh')
        body = None
        points = None
        point_collect = []
        for key in decimated.keys():
            print(key)
            point = hv.Scatter(decimated[key], SSC, FSC, label=key)
            point_collect.append(point)
            if points is None:
                points = point
            else:
                points *= point
        if body is None:
            body = points.opts(title='Default {0}: SSC vs FSC'.format("Combined"), height=450, width=450)
        else:
            body += points.opts(title='Default {0}: SSC vs FSC'.format("Combined"))

        for dim in (SSC, FSC):
            hists = None
            for point in point_collect:
                hist = histogram(point, dimension=dim)
                if hists is None:
                    hists = hist
                else:
                    hists *= hist
            body += hists

        potentialCols = [c for c in decimated[list(decimated.keys())[0]].columns if c != SSC and c != FSC]
        for i in range(len(potentialCols)):
            for j in range(i+1, len(potentialCols)):
                points = None
                point_collect = []
                for key in decimated.keys():
                    point = hv.Scatter(decimated[key], potentialCols[i], potentialCols[j], label=key)
                    point_collect.append(point)
                    if points is None:
                        points = point
                    else:
                        points *= point
                body += points.opts(title='Combined: {0} vs {1}'.format(potentialCols[i], potentialCols[j]), height=450, width=450)

                for dim in (potentialCols[i], potentialCols[j]):
                    hists = None
                    for point in point_collect:
                        hist = histogram(point, dimension=dim)
                        if hists is None:
                            hists = hist
                        else:
                            hists *= hist
                    body += hists
        body = body.opts(
            opts.Scatter(alpha=0.9),
            opts.Histogram(alpha=0.9, height=450),
            opts.Layout(shared_axes=True, shared_datasource=True)).cols(3)
        renderer.save(body, os.path.join(self.directory, cachebust+"combined_gating"))

    def coordinate_gating(self, df, col1, col2, xlimit, ylimit, type):
        renderer = hv.renderer('bokeh')
        d = {col1: df[col1], col2: df[col2], "category": 0}
        xycols = pd.DataFrame(data=d)

        set1 = []
        set2 = []
        set3 = []
        set4 = []
        for index in range(len(xycols)):
            x = float(xycols[col1][index])
            y = float(xycols[col2][index])
            xlimit = float(xlimit)
            ylimit = float(ylimit)
            if x < xlimit and y > ylimit:
                #xycols["category"][index] = 1
                set1.append((xycols[col1][index], xycols[col2][index]))
            elif x > xlimit and y > ylimit:
                #xycols["category"][index] = 2
                set2.append((xycols[col1][index], xycols[col2][index]))
            elif x < xlimit and y < ylimit:
                #xycols["category"][index] = 3
                set3.append((xycols[col1][index], xycols[col2][index]))
            elif x > xlimit and y < ylimit:
                #xycols["category"][index] = 4
                set4.append((xycols[col1][index], xycols[col2][index]))

        categories = {}
        categories["upper left"] = np.around(100*len(set1)/len(xycols), decimals=2)
        categories["upper right"] = np.around(100*len(set2)/len(xycols), decimals=2)
        categories["bottom left"] = np.around(100*len(set3)/len(xycols), decimals=2)
        categories["bottom right"] = np.around(100*len(set4)/len(xycols), decimals=2)

        linex = hv.VLine(xlimit)
        liney = hv.HLine(ylimit)
        body = linex * liney * hv.Points(set1, label=str("upper left: " + str(categories["upper left"]) + "%")) * \
               hv.Points(set2, label=str("upper right: " + str(categories["upper right"]) + "%")) * \
               hv.Points(set3, label=str("bottom left: " + str(categories["bottom left"]) + "%")) * \
               hv.Points(set4, label=str("bottom right: " + str(categories["bottom right"]) + "%"))
        body = body.opts(plot=dict(width=800, height=600))
        body = body.opts(opts.Points(tools=['box_select', 'lasso_select', 'hover'], size=6, fill_alpha=0.6))
        body = body.redim.label(x=col1, y=col2)
        if os.path.isfile(os.path.join(self.directory, str("coordinate_gating_"+str(type)))):
            os.remove(os.path.join(self.directory, str("coordinate_gating_"+str(type))))
        renderer.save(body, os.path.join(self.directory, str("coordinate_gating_"+str(type))))

    


def initiateAnalysis(directory, selected_cols, available_cols, files):
    print("initiate")
    sqlexp = SQLVis(directory, selected_cols, available_cols, files)
    sqlexp.parseSQL(files)
    print("kmeans")
    sqlexp.num_clusters = min(max(250, len(sqlexp.exps[files[0]])//40), 5000)
    sqlexp.decimateKMeans(sqlexp.num_clusters)
    print('before db intitate')
    sqlexp.sqldb_kmeans()
    print("after kmeans")
    return sqlexp


def generateHTML(directory, selected_cols, available_cols, files, option, selected=[0,1], xlimit=0, ylimit=0, cachebust="",
                 redir_url=None, noChange=True):
    sqlexp = SQLVis(directory, selected_cols, available_cols, files)
    decimated = sqlexp.parseSQL(["kmeans_{0}".format(f) for f in files])
    (SSC, FSC) = sqlexp.findColSSCFSC()
    jobs = []


    #option 1 or 2
    if option in range(len(files)):
        for key in decimated.keys():
            p = Process(target=sqlexp.generateHoloview,
                                        args=(decimated[key], SSC, FSC, str(cachebust+key)))
            jobs.append(p)
            p.start()

    #option 3
    if option == len(files):
        task = Process(target=sqlexp.generateCombined,
                                       args =(decimated, SSC, FSC, cachebust))
        jobs.append(task)
        task.start()

    #option 4
    if option == len(files) + 1:
        col1 = sqlexp.all_cols[selected_cols[selected[0]]]
        col2 = sqlexp.all_cols[selected_cols[selected[1]]]
        print(col1, col2)
        if xlimit == 0 and ylimit == 0:
            xlimit = np.around(np.median(decimated[list(decimated.keys())[0]][col1]), decimals=3)
            ylimit = np.around(np.median(decimated[list(decimated.keys())[0]][col2]), decimals=3)
        for key in decimated.keys():
            p = Process(target=sqlexp.coordinate_gating,
                                        args = (decimated[key], col1, col2, xlimit, ylimit, str(key+cachebust)))
            jobs.append(p)
            p.start()

    #option 5
    if option == len(files) + 2:
        import gating
        if redir_url is not None:
            if noChange:
                index = 0
                for key in decimated.keys():
                    gating.constructLines(directory, decimated[key], SSC, FSC, cachebust, redir_url[files[index]], file=files[index])
                    index += 1
            else:
                index = 0
                for key in decimated.keys():
                    col1 = sqlexp.all_cols[selected_cols[selected[0]]]
                    col2 = sqlexp.all_cols[selected_cols[selected[1]]]
                    gating.constructLines(directory, decimated[key], col1, col2, cachebust, redir_url[files[index]], file=files[index])
                    index += 1
    for proc in jobs:
        proc.join()

    return xlimit, ylimit
