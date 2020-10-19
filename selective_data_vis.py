import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cytoflow as flow
import seaborn as sns
import os
from scipy.stats import ttest_ind
import sqlite3
import flowutils as fu
from multiprocessing import Process
from multiprocessing import Manager
from pandas.plotting import scatter_matrix
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from scipy.stats import gaussian_kde

class Experiment:
    """

    Wrapper class that holds all the data processing, analysis, and visualization functions for a standard flow
    cytometry analysis.

    Class is constructed every time a new job is created, not for individual experimental files that are uploaded.

    """
    def __init__(self, directory, transformation, file_loc, tSNE_O, KDE):
        """
        Initialize basic settings for the experimental files uploaded, including the size of images generated, and
        calls to initiate the SQLite3 database to store the experimental data.

        self.exps : Sets up a data dictionary, whose keys represent individual files that were uploaded.
        self.columns : Sets up a columns list, which includes all parsed, possible flow cytometry fluoresence channels
        that can be visualized
        self.memoizeMins : Sets up a dictionary that contains parameters for histogram graphing, specifically the file
        and column, and what the minimum values to graph that (based on distribution statistics) will be. An example
        entry is (file, col) -> 141
        self.memoizeMax : Sets up a dictionary that contains parameters for the maximum values to plot

        Parameters
        ----------
        directory : str
            Full path directory to the experimental folder (ie. CUR_DIR/Date/int_hash)
        transformation : str
            The transformation option (ie. raw, logicle, hyperlog) selected when the files were uploaded.
        file_loc : str
            Full path to the data folder, where the raw FCS files are stored (ie. CUR_DIR/Date)

        Returns
        -------
        int
            Description of return value

        """
        self.directory = directory
        self.transformation = transformation
        self.file_loc = file_loc
        self.exps = {}
        self.columns = []
        plt.rcParams['figure.figsize'] = [8, 5]
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams['axes.labelsize'] = 12
        self.memoizeMins = {}
        self.memoizeMax = {}
        self.tSNE_O = tSNE_O
        self.KDE = KDE
        self.initDB()

    def initDB(self):
        """
        Initialize the SQL database to store the experimental data, creating the file "experiment.db".

        self.conn : Attempts to make a connection to the database in self.directory, and attempts to store that
        connection.

        Parameters
        ----------

        Returns
        -------

        """
        db_file = os.path.join(self.directory, "experiment.db")
        print(db_file)
        try:
            self.conn = sqlite3.connect(db_file)
        except:
            pass

    def getColNames(self, cols):
        """
        Get the names of all the valid columns that can be visualized.

        Parameters
        ----------
        cols : list(int)
        A list of user-inputted columns (based on index) to visualized.

        Returns
        -------
        list(str)
            A list of the columns full names, based on user-selected input, to be visualized.

        """
        return [self.columns[i] for i in cols]

    def _init(self, expFiles):
        """
        Wrapper function that initializes data processing, converting the raw FCS files into pandas dataframes.

        Utilizes multiprocessing to allow processing of the FCS files in parallel, goal is to speed up this process,
        which can take a long time if the experiment contains millions of events and/or an expensive transformation is
        applied to the dataset.

        self.exps : Sets the value of this dictionary to be "filename" -> pandasDF
        self.columms : Sets the value of this list to be all the columns that can possibly be analyzed. This means
        that the parameter for future machine learning training options ("NN") and the parameter for time ("Time") are
        eliminated at this time.

        Parameters
        ----------
        expFiles : list(str)
        A list of all the names of the FCS files the user has uploaded.

        """
        manager = Manager()
        return_dict = manager.dict()
        col_list = manager.list()
        col_list.extend(self.columns)

        def worker(file, return_dict, cols, condition=0):
            """
            Calls the function self.createExperiment through multiprocessing jobs.

            Parameters
            ----------
            file : str
                Name of the FCS file.
            return_dict : Manager().dict() or {}
                A way to share data across multiprocessing jobs. Can convert to regular python dictionary by invoking
                dict function.
            cols : Manager().list() or list(str)
                Share the names of columns across multiprocessing jobs. The goal is for this list to be the columns that
                all the experimental files uploaded share.

            """
            data = self.createExperiment(file, condition)
            return_dict[file] = data
            if len(cols) == 0:
                cols.extend([c for c in data.columns if c is not "NN" and c is not "Time"])
            else:
                for index in range(len(cols)):
                    if cols[index] not in data.columns:
                        cols.pop(index)

        jobs = []
        for file in expFiles:
            p = Process(target=worker, args=(file, return_dict, col_list))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        print(self.exps, self.columns)
        self.exps = dict(return_dict)
        self.columns = list(col_list)

    def createExperiment(self, fcsfile, condition):
        """
        Data processing for an individual FCS file, allowing for data transformations to be applied at this time.

        Reads the data through the CytoFlow API,
        Converts the data into a numPy dataset, applies transformations as requested through the Flow Utils API,
        and then converts it back into a pandas Dataframe, adding it to the SQL database file in the process.
        This step can be optimized in the future to reduce memory and increase the speed of transformations.

        Parameters
        ----------
        fcsfile : str
            Name of the FCS file to process
        condition : int
            Value of a classificiation column for future potential machine learning implementations.

        """
        tube1 = flow.Tube(file = os.path.join(self.file_loc, fcsfile),
                          conditions = {'NN' : condition})
        import_op = flow.ImportOp(conditions = {'NN' : 'float'},
                                  tubes = [tube1])

        exp = import_op.apply()
        allDat = exp.data

        def convertToDF(nparr, columns):
            data = pd.DataFrame(columns=columns)
            for i in range(len(nparr)):
                data.loc[i] = list(nparr[i])
            return data

        def hyperlog(df):
            nparr = df.values
            mod_events = fu.transforms.hyperlog(nparr, np.arange(0, len(df.columns)))
            mod_pd = convertToDF(mod_events, df.columns)
            return mod_pd
        def logicle(df):
            nparr = df.values
            mod_events = fu.transforms.logicle(nparr, np.arange(0, len(df.columns)))
            mod_pd = convertToDF(mod_events, df.columns)
            return mod_pd

        if(self.transformation == "hyperlog"):
            allDat = hyperlog(allDat)
        elif(self.transformation == "logicle"):
            allDat = logicle(allDat)

        #Export to SQLDB
        try:
            allDat.to_sql(fcsfile, self.conn)
        except Error as e:
            print(e)
        return allDat

    def manageVisualizations(self, channels, files):
        """
        Multiprocessing wrapper function to perform the functions needed to visualize the dataset.

        Tasks to create scatterplots, boxwhisker diagrams, scatter matrix, combined box-whisker plots, histograms,
        correlation heatmaps, and generate an Excel Spreadsheet file are performed at this step in parallel.

        Parameters
        ----------
        channels : list(str)
            List of the channels that should be visualized
        files : list(str)
            List of the names of the FCS files that should be visualized.

        """
        tasks = set()
        if self.tSNE_O:
            tasks.add(Process(target=self.tSNE(files, 50)))
            #tasks.add(Process(target=self.tSNE(files, 30)))
        if self.KDE:
            tasks.add(Process(target=self.gaussianKDEPlot, args=(channels, files)))
        tasks.add(Process(target=self.scatter, args=(channels, files)))
        tasks.add(Process(target=self.histogram, args=(channels, files)))
        tasks.add(Process(target=self.boxwhisker, args=(channels, files)))
        tasks.add(Process(target=self.scatterMatrix, args=(channels, files)))
        tasks.add(Process(target=self.combinedboxwhisker, args=(channels, files)))
        tasks.add(Process(target=self.pairwiseCorrelationHeatmap, args=(channels, files)))
        tasks.add(Process(target=self.getXLSX, args=(channels, files)))

        for task in tasks:
            task.start()
        for task in tasks:
            task.join()

    def graphMin(self, cols, files, level="5%"):
        """
        Identifies the lower bound for graphing in case of outliers.

        Parameters
        ----------
        cols : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.
        level : str
            Distribution percentile to obtain data from.

        Returns
        -------
        int
            minimum value, based on level inputted, to serve as the lower bound for visualizations

        """
        min_val = float('inf')
        for col in cols:
            for file in files:
                if (file, col) not in self.memoizeMins:
                    self.memoizeMins[(file, col)] = self.exps[file][col].describe(percentiles=[.05, 0.07, 0.1])[level]
                min_val = min(min_val, self.memoizeMins[(file, col)])
        return max(min_val, 0)

    def graphMax(self, cols, files, level="75%"):
        """
        Identifies the upper bound for graphing in case of outliers.

        Parameters
        ----------
        cols : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.
        level : str
            Distribution percentile to obtain data from.

        Returns
        -------
        int
            maximum value, based on level inputted, to serve as the upper bound for visualizations

        """
        max_val = float('-inf')
        for col in cols:
            for file in files:
                if (file, col) not in self.memoizeMax:
                    self.memoizeMax[(file, col)] = self.exps[file][col].describe(percentiles=[0.75, 0.9, 0.95])[level]
                max_val = max(max_val, self.memoizeMax[(file, col)])
        return max_val

    def scatter(self, channels, files):
        """
        Generates scatterplots based on the channels selected, and the files inputted.

        The number of scatterplots generated == len(channels) * (len(channels)-1)
        The scale of the plots are all linear
        The scatterplots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        for i in range(len(channels)):
            for j in range(i+1, len(channels)):
                plots = []
                coli, colj = channels[i], channels[j]
                bin_min, bin_max = 0, float('-inf')
                for file in files:
                    plots.append(plt.scatter(self.exps[file][coli].values, self.exps[file][colj].values, alpha=0.5, s=np.sqrt(12), label="file"))
                title = "{0} \nScatterplot: {1} vs {2}".format(str(files), coli, colj)
                plt.title(title)
                plt.xlabel(coli)
                plt.ylabel(colj)
                plt.legend(plots, files, loc="best")
                plt.yscale("linear")
                plt.xscale("linear")
                bin_min = self.graphMin([coli, colj], files)
                plt.xlim(bin_min, self.graphMax([coli], files, level="95%")*1.2)
                plt.ylim(bin_min, self.graphMax([colj], files, level="95%")*1.2)
                plt.savefig(os.path.join(self.directory, title+".png"))
                plt.clf()

    def boxwhisker(self, channels, files):
        """
        Generates boxwhisker diagrams based on the channels selected, and the files inputted.

        The number of box whisker diagrams generated == len(files)
        The box whisker diagrams depict the channels selected for every file uploaded.
        The scale of the plots are logistic (symlog to allow graphing even if values are negative).
        The plots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        for file in files:
            self.exps[file].boxplot(column=channels, rot=0)
            max_y = self.graphMax(channels, [file])
            min_y = self.graphMin(channels, [file])
            if len(channels) <= 5:
                title = "{0} \nboxplot for: {1}".format(str(file), str(channels))
            else:
                title = "{0} \nboxplot".format(str(file))
            plt.title(title)
            plt.ylabel("Fluorescence Value")
            plt.yscale("symlog")
            plt.ylim(min_y, max_y*1.1)
            plt.savefig(os.path.join(self.directory, title+".png"))
            plt.clf()

    def scatterMatrix(self, channels, files):
        """
        Generates scatterMatrices based on the channels selected, and the files inputted.

        The number of plots generated == len(files)
        The scale of the plots are all logistic
        The plots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        for file in files:
            data = self.exps[file][channels]
            scatter_matrix(data, alpha=0.2, figsize=(len(channels)*4, len(channels)*4), diagonal='kde')
            if len(channels) <= 5:
                title = "{0} \ndistribution overview (scatter matrix) for: {1}".format(str(file), str(channels))
            else:
                title = "{0} \ndistribution overview (scatter matrix)".format(str(file))
            plt.yscale("symlog")
            plt.xscale("symlog")
            plt.savefig(os.path.join(self.directory, title+".png"))
            plt.clf()

    def combinedboxwhisker(self, channels, files):
        """
        Generates a combined boxwhisker diagram based on the channels selected, and the files inputted.
        Generates a stats.txt file which is the result of t-test run on every channel between every file.

        This combined diagram depicts all the files side by side, seperated by every channel in channels.

        The number of combined boxwhisker diagrams generated == 1
        The scale of the plot is logistical
        The plots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluorescence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        frames = {}
        for file in files:
            frames[file] = self.exps[file][channels]
        comb = pd.concat(frames, axis=1)
        comb.columns = comb.columns.swaplevel(0, 1)
        comb.sort_index(axis=1, level=0, inplace=True)
        fig, ax = plt.subplots()
        size = 4*len(channels)*len(files)
        comb.mean().plot.bar(yerr=comb.std(), ax=ax, capsize=4, figsize=(size,20), colormap='Paired')
        bin_min = self.graphMin(channels, files)
        plt.ylabel("Fluorescence Value")
        plt.yscale("symlog")
        plt.ylim(bin_min)
        if len(channels) <= 5:
            title = "{0} \nCombined Boxplot: {1}".format(files, channels)
        else:
            title = "{0} \nCombined Boxplot".format(files)
        plt.title(title)
        file = open(os.path.join(self.directory, "stats.txt"),"w")
        for x in range(len(files)):
            for y in range(x+1, len(files)):
                for channel in channels:
                    file.write("{0} vs {1}: {2} | ".format(files[x], files[y], channel) +
                               str(self.ttest(self.exps[files[x]][channel], self.exps[files[y]][channel])) + "} \n")
        file.close()
        plt.savefig(os.path.join(self.directory, title+".png"))
        plt.clf()

    def pairwiseCorrelationHeatmap(self, channels, files):
        """
        Generates pairwise correlation heatmaps based on the channels selected, and the files inputted.

        Each pairwise correlation shows how related one channel is to another on the same file (1 = maximum correlation,
        0 = no correlation, -1 = inverse relationship)

        The number of plots generated == len(files)
        The plots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        for file in files:
            data = self.exps[file][channels]
            fig,ax = plt.subplots(figsize=(8,5))
            map = sns.heatmap(data.corr(), vmin=0, vmax=1, ax=ax,
                        square=True, annot=True, linewidths=0.05, fmt= '.2f',cmap="twilight_shifted")
            map.set_xticklabels(map.get_xticklabels(), rotation=0)
            map.set_yticklabels(map.get_yticklabels(), rotation=30)
            if len(channels) <= 5:
                title = "{0} \nFluorescent Channels Pairwise Correlation for: \n{1}".format(file, channels)
            else:
                title = "{0} \nFluorescent Channels Pairwise Correlation".format(file)
            plt.title(title)
            plt.savefig(os.path.join(self.directory, title+".png"))
            plt.clf()

    def histogram(self, channels, files):
        """
        Generates histograms based on the channels selected, and the files inputted.
        Generates describe.txt file for every file.

        The number of histograms generated == len(files)
        The y scale of the plot is linear (frequency) and the x scale (fluorescence) is logistical
        The plots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.

        """

        f = open(os.path.join(self.directory, "describe.txt"),"a")
        for channel in channels:
            bin_max = self.graphMax([channel], files, level="90%")
            bin_min = self.graphMin([channel], files)
            plt.xlim(bin_min, bin_max)

            for file in files:
                data = self.exps[file][channel]
                binsnp = np.linspace(self.graphMin([channel], [file]), self.graphMax([channel], [file]), 50)
                n, bins, patches = plt.hist(data, bins=binsnp, alpha=0.7, label=file)
                f.write("{" + "{0} {1}".format(file, channel) + " : " + str(self.statistics(data)) + "}")
                #(mu_control, sigma_control) = scipy.stats.norm.fit(controlCols[index])
                #y = scipy.stats.norm.pdf(bins, mu_control, sigma_control)
                #plt.plot(bins, y, '--', label='Control Normal PDF', color='blue')
            plt.legend(loc='best')
            title = "Histogram of {0} for {1}".format(channel, files)
            plt.title(title)
            plt.ylabel("Frequency")
            plt.ylim(0)
            plt.xlabel("Fluorescence Value")
            plt.yscale("linear")
            plt.xscale("linear")
            plt.savefig(os.path.join(self.directory, title+".png"))
            plt.clf()
        f.close()

    def gaussianKDEPlot(self, channels, files):
        """
        Generates a KDE heatmap scatterplot based on the channels selected, and the files inputted.

        The number of plots generated == len(files)
        The plots are saved in self.directory, and are uniquely identified based on title (same as on the graph).

        Parameters
        ----------
        channels : list(str)
            List of fluoresence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                plots = []
                coli, colj = channels[i], channels[j]
                bin_min, bin_max = 0, float('-inf')
                for file in files:
                    x = self.exps[file][coli].values
                    y = self.exps[file][colj].values
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    a = plt.scatter(x, y, c=z, s=12, cmap="twilight_shifted", label=file)
                    plt.colorbar(a, ticks=[0], orientation="vertical")
                    plots.append(a)
                    title = "{0} \nKDE Heatmap scatterplot: \n{1} vs {2}".format(str(file), coli, colj)
                    plt.title(title)
                    plt.xlabel(coli)
                    plt.ylabel(colj)
                    plt.legend(loc="best")
                    plt.yscale("linear")
                    plt.xscale("linear")
                    bin_min = self.graphMin([coli, colj], files)
                    plt.xlim(bin_min, self.graphMax([coli], files, level="95%") * 1.2)
                    plt.ylim(bin_min, self.graphMax([colj], files, level="95%") * 1.2)
                    plt.savefig(os.path.join(self.directory, title + ".png"))
                    plt.clf()

    def tSNE(self, files, perplexity):
        n_components = 2

        fig, ax = plt.subplots()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        title = "t-SNE (Perplexity = {0}) of all channels for \n{1}".format(perplexity, files)
        ax.set_title(title)
        plots = []
        X, lengths = [], []
        for file in files:
            X += [self.exps[file].values]
            lengths.append(len(self.exps[file]))
        X = np.concatenate(X)
        tsneFit = tsne.fit_transform(X)
        lastIndex = 0
        for i in range(len(lengths)):
            plots.append(ax.scatter(tsneFit[lastIndex:lengths[i],0], tsneFit[lastIndex:lengths[i], 1], alpha=0.5, s=np.sqrt(12)))
        plt.legend(plots, files, loc="best")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        plt.savefig(os.path.join(self.directory, title+".png"))
        plt.clf()

    def getXLSX(self, channels, files):
        """
        Generates an excel Spreadsheet file with every channel selected, and file inputted (every page = a file)

        The number of excel pages generated == len(files)
        The excel file is saved in self.directory, saved as "alldata.xlsx"

        Parameters
        ----------
        channels : list(str)
            List of fluorescence channels to analyze.
        files : list(str)
            List of files to analyze.

        """
        save_loc = os.path.join(self.directory, "alldata.xlsx")
        writer = pd.ExcelWriter(save_loc, engine='xlsxwriter')
        for file in files:
            data = self.exps[file][channels]
            data.to_excel(writer, sheet_name=file)
        writer.save()

    def ttest(self, col1, col2):
        """
        Invokes sciPy's ttest for independence on 2 channels.

        Parameters
        ----------
        col1 : str
            1st Fluorescence channel to analyze
        col2 : str
            2nd Fluorescence channel to analyze

        """
        return ttest_ind(col1, col2)
    def statistics(self, col_data):
        """
        Invokes pandas's describe method to generate distribution statistics for every experiment

        Parameters
        ----------
        col_data : pandas Series
            Single column in pandas Dataframe to generate distribution statistics

        """
        return str(col_data.describe())


def constructExperiment(directory, expFiles, transformation, file_loc, tSNE_O=False, KDE=False):
    """
    Wrapper function that constructs the Experiment object, and initializes its values (without further user input
    to select columns).

    Parameters
    ----------
    directory : str
        Full path directory to the experimental folder (ie. CUR_DIR/Date/int_hash)
    expFiles : list(str)
        List of all experimental FCS files.
    transformation : str
        The transformation option (ie. raw, logicle, hyperlog) selected when the files were uploaded.
    file_loc : str
        Full path to the data folder, where the raw FCS files are stored (ie. CUR_DIR/Date)
    tSNE_O : boolean
        Option to use tSNE analysis or not (computationally intensive)
    Returns
    -------
    Experiment
        instance of Experiment class with the initialized values.

    """
    a = Experiment(directory, transformation, file_loc, tSNE_O, KDE)
    a._init(expFiles)
    return a

def performDataAnalyses(expObj, channels, files):
    """
    Wrapper function that constructs the Experiment object, and initializes its values (without further user input
    to select columns).

    Parameters
    ----------
    expObj : Experiment
        Initialized Experiment object (has data).
    channels : list(str)
        List of all selected fluorescence channels to visualize.
    files : list(str)
        List of all selected experimental FCS files to visualize.

    """
    channels = expObj.getColNames(channels)
    expObj.manageVisualizations(channels, files)
