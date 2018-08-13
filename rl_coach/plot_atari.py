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

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from rl_coach.dashboard_components.signals_file import SignalsFile


class FigureMaker(object):
    def __init__(self, path, cols, smoothness, signal_to_plot, x_axis, color):
        self.experiments_path = path
        self.environments = self.list_environments()
        self.cols = cols
        self.rows = int((len(self.environments) + cols - 1) / cols)
        self.smoothness = smoothness
        self.signal_to_plot = signal_to_plot
        self.x_axis = x_axis
        self.color = color

        params = {
            'axes.labelsize': 8,
            'font.size': 10,
            'legend.fontsize': 14,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.usetex': False,
            'figure.figsize': [16, 30]
        }
        matplotlib.rcParams.update(params)

    def list_environments(self):
        environments = sorted([e.name for e in os.scandir(self.experiments_path) if e.is_dir()])
        filtered_environments = self.filter_environments(environments)
        return filtered_environments

    def filter_environments(self, environments):
        filtered_environments = []
        for idx, environment in enumerate(environments):
            path = os.path.join(self.experiments_path, environment)
            experiments = [e.name for e in os.scandir(path) if e.is_dir()]

            # take only the last updated experiment directory
            last_experiment_dir = max([os.path.join(path, root) for root in experiments], key=os.path.getctime)

            # make sure there is a csv file inside it
            for file_path in os.listdir(last_experiment_dir):
                full_file_path = os.path.join(last_experiment_dir, file_path)
                if os.path.isfile(full_file_path) and file_path.endswith('.csv'):
                    filtered_environments.append((environment, full_file_path))

        return filtered_environments

    def plot_figures(self, prev_subplot_map=None):
        subplot_map = {}
        for idx, (environment, full_file_path) in enumerate(self.environments):
            environment = environment.split('level')[1].split('-')[1].split('Deterministic')[0][1:]
            if prev_subplot_map:
                # skip on environments which were not plotted before
                if environment not in prev_subplot_map.keys():
                    continue
                subplot_idx = prev_subplot_map[environment]
            else:
                subplot_idx = idx + 1
            print(environment)
            axis = plt.subplot(self.rows, self.cols, subplot_idx)
            subplot_map[environment] = subplot_idx
            signals = SignalsFile(full_file_path)
            signals.change_averaging_window(self.smoothness, force=True, signals=[self.signal_to_plot])
            steps = signals.bokeh_source.data[self.x_axis]
            rewards = signals.bokeh_source.data[self.signal_to_plot]

            yloc = plt.MaxNLocator(4)
            axis.yaxis.set_major_locator(yloc)
            axis.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.title(environment, fontsize=10, y=1.08)
            plt.plot(steps, rewards, self.color, linewidth=0.8)
            plt.subplots_adjust(hspace=2.0, wspace=0.4)

        return subplot_map

    def save_pdf(self, name):
        plt.savefig(name + ".pdf", bbox_inches='tight')

    def show_figures(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths',
                        help="(string) Root directory of the experiments",
                        default=None,
                        type=str)
    parser.add_argument('-c', '--cols',
                        help="(int) Number of plot columns",
                        default=6,
                        type=int)
    parser.add_argument('-s', '--smoothness',
                        help="(int) Number of consequent episodes to average over",
                        default=100,
                        type=int)
    parser.add_argument('-sig', '--signal',
                        help="(str) The name of the signal to plot",
                        default='Evaluation Reward',
                        type=str)
    parser.add_argument('-x', '--x_axis',
                        help="(str) The meaning of the x axis",
                        default='Total steps',
                        type=str)
    parser.add_argument('-pdf', '--pdf',
                        help="(str) A name of a pdf to save to",
                        default='atari',
                        type=str)
    args = parser.parse_args()

    paths = args.paths.split(",")
    subplot_map = None
    for idx, path in enumerate(paths):
        maker = FigureMaker(path, cols=args.cols, smoothness=args.smoothness, signal_to_plot=args.signal, x_axis=args.x_axis, color='C{}'.format(idx))
        subplot_map = maker.plot_figures(subplot_map)
    plt.legend(paths)
    maker.save_pdf(args.pdf)
    maker.show_figures()
