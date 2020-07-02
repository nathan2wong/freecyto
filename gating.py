###Bokeh
from bokeh.models import LassoSelectTool, BoxSelectTool, HoverTool, ColumnDataSource, FreehandDrawTool, WheelZoomTool, Range1d, Button
from bokeh.models.callbacks import CustomJS
from bokeh.io import save
from bokeh.plotting import figure, output_file, show, reset_output
from bokeh.layouts import widgetbox, gridplot, layout
import numpy as np
import pandas as pd
import os

def constructLines(directory, df, col1, col2, cachebust, redir_url, file):
    filename = str(file+"gating"+cachebust+".html")

    output_file(os.path.join(directory, filename))
    print(df.columns)
    x = df[col1]
    y = df[col2]

    data = pd.DataFrame(dict(x=x, y=y))

    source = ColumnDataSource(data)

    handler = CustomJS(args=dict(source=source), code="""
    var src_dat = source.selected['1d']['indices'];
    var conv_data = JSON.stringify(src_dat);
    console.log(conv_data);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", '{0}', true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.send("json=" + conv_data);
    var but = document.getElementsByClassName("bk-btn-success").item(0);
    but.innerHTML = "<p>Reload the page (auto reload in 5 seconds)</p>";
    console.log(but);""".format(redir_url, file))

    p = figure(plot_height = 500, plot_width = 1000,
               title = col1 + " vs " + col2 + ' Gating',
               x_axis_label = col1, y_axis_label = col2)
    p.scatter(x="x", y="y", source=source, fill_alpha=0.6, size=8)


    hover = HoverTool(tooltips = [(col1, "$x"), (col2, "$y")])

    p.add_tools(hover)
    p.add_tools(LassoSelectTool())
    p.add_tools(BoxSelectTool())

    bargraphs = []
    bargraphs.append(p)
    btn = Button(label='Submit selected points', button_type="success", callback=handler)
    bargraphs.append(btn)
    #df1[c[activegroup.data['source']]], df2[c[activegroup.data['source']]]
    #grid = gridplot([widgetbox(view_selection)], bargraphs, scatterplots)
    l = layout(bargraphs)
    save(l)