import argparse
import matplotlib
import matplotlib.pyplot as plt
from dashboard import SignalsFile
import os


class FigureMaker(object):
    def __init__(self, path, cols, smoothness, signal_to_plot, x_axis):
        self.experiments_path = path
        self.environments = self.list_environments()
        self.cols = cols
        self.rows = int((len(self.environments) + cols - 1) / cols)
        self.smoothness = smoothness
        self.signal_to_plot = signal_to_plot
        self.x_axis = x_axis

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
        environments = sorted([e.name for e in os.scandir(args.path) if e.is_dir()])
        filtered_environments = self.filter_environments(environments)
        return filtered_environments

    def filter_environments(self, environments):
        filtered_environments = []
        for idx, environment in enumerate(environments):
            path = os.path.join(args.path, environment)
            experiments = [e.name for e in os.scandir(path) if e.is_dir()]

            # take only the last updated experiment directory
            last_experiment_dir = max([os.path.join(path, root) for root in experiments], key=os.path.getctime)

            # make sure there is a csv file inside it
            for file_path in os.listdir(last_experiment_dir):
                full_file_path = os.path.join(last_experiment_dir, file_path)
                if os.path.isfile(full_file_path) and file_path.endswith('.csv'):
                    filtered_environments.append((environment, full_file_path))

        return filtered_environments

    def plot_figures(self):
        for idx, (environment, full_file_path) in enumerate(self.environments):
            print(environment)
            axis = plt.subplot(self.rows, self.cols, idx + 1)
            signals = SignalsFile(full_file_path)
            signals.change_averaging_window(self.smoothness, force=True, signals=[self.signal_to_plot])
            steps = signals.bokeh_source.data[self.x_axis]
            rewards = signals.bokeh_source.data[self.signal_to_plot]

            yloc = plt.MaxNLocator(4)
            axis.yaxis.set_major_locator(yloc)
            axis.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.title(environment, fontsize=10, y=1.08)
            plt.plot(steps, rewards, linewidth=0.8)
            plt.subplots_adjust(hspace=2.0, wspace=0.4)

    def save_pdf(self, name):
        plt.savefig(name + ".pdf", bbox_inches='tight')

    def show_figures(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
                        help="(string) Root directory of the experiments",
                        default=None,
                        type=str)
    parser.add_argument('-c', '--cols',
                        help="(int) Number of plot columns",
                        default=6,
                        type=int)
    parser.add_argument('-s', '--smoothness',
                        help="(int) Number of consequent episodes to average over",
                        default=200,
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

    maker = FigureMaker(args.path, cols=args.cols, smoothness=args.smoothness, signal_to_plot=args.signal, x_axis=args.x_axis)
    maker.plot_figures()
    maker.save_pdf(args.pdf)
    maker.show_figures()
