#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from bokeh.layouts import row, column
from bokeh.models.widgets import Div

from rl_coach.dashboard_components.experiment_board import file_selection_button, group_selection_button
from rl_coach.dashboard_components.globals import layouts

# title
title = Div(text="""<h1>Coach Dashboard</h1>""")

# landing page
landing_page_description = Div(text="""<h3>Start by selecting an experiment file or directory to open:</h3>""")
center = Div(text="""<style>html { text-align: center; } </style>""")
center_buttons = Div(text="""<style>.bk-root .bk-widget { margin: 0 auto; }</style>""", width=0)
landing_page = column(center,
                      title,
                      landing_page_description,
                      row(center_buttons),
                      row(file_selection_button, sizing_mode='scale_width'),
                      row(group_selection_button, sizing_mode='scale_width'),
                      sizing_mode='scale_width')

layouts['landing_page'] = landing_page
