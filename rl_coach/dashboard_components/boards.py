from bokeh.layouts import column
from bokeh.models.widgets import Panel, Tabs
from rl_coach.dashboard_components.experiment_board import experiment_board_layout
from rl_coach.dashboard_components.episodic_board import episodic_board_layout
from rl_coach.dashboard_components.globals import spinner, layouts
from bokeh.models.widgets import Div

# ---------------- Build Website Layout -------------------

# title
title = Div(text="""<h1>Coach Dashboard</h1>""")
center = Div(text="""<style>html { padding-left: 50px; } </style>""")
tab1 = Panel(child=experiment_board_layout, title='experiment board')
# tab2 = Panel(child=episodic_board_layout, title='episodic board')
# tabs = Tabs(tabs=[tab1, tab2])
tabs = Tabs(tabs=[tab1])

layout = column(title, center, tabs)
layout = column(layout, spinner)

layouts['boards'] = layout
