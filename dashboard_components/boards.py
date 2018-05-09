from bokeh.layouts import column
from bokeh.models.widgets import Panel, Tabs
from dashboard_components.experiment_board import experiment_board_layout
from dashboard_components.globals import spinner, layouts
from bokeh.models.widgets import Div

# ---------------- Build Website Layout -------------------

# title
title = Div(text="""<h1>Coach Dashboard</h1>""")

tab1 = Panel(child=experiment_board_layout, title='experiment board')
tabs = Tabs(tabs=[tab1])

layout = column(title, tabs)
layout = column(layout, spinner)

layouts['boards'] = layout
