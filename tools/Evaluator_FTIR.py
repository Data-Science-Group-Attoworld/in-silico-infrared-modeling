import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .calculation_functions import *
from .calculation_functions import _NoneScaler
from .Evaluator import *
from .Loss_Metric_Manager import *


class Evaluator_FTIR(Evaluator):
    '''
    Inherits all calculations (i.e. add_ functions) from Evaluator. 
    Only the plotting functions for spectroscopy data are changed. 
    These plots now remove the silent region via broken axis plots.
    '''


    def __init__(self, spectra_map, scaler = None, loss_metric_manager=None, vec=None, init_figure_settings_once=False):
        super().__init__(
            spectra_map=spectra_map,
            scaler=scaler,
            loss_metric_manager=loss_metric_manager,
            vec=vec,
            init_figure_settings_once=init_figure_settings_once)


    class Dist(Evaluator.Dist):
        def __init__(self, parent):
            super().__init__(parent)

            self.wavenumber_index_to_look_at = -9999

        def plot_sample_spectrum(self, save_fig=True):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            # Plot spectrum
            for stype in self.sample_spectrum.keys():
                ax1.plot(x, self.sample_spectrum[stype], label=f'{stype}', lw=0.5, color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, self.sample_spectrum[stype], lw=0.5, color=self.parent.color_gen.get_color(stype))

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)

            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            title = f'One sample spectrum {self.parent.title_addition}'
            ax1.set_title(title, fontsize=14)
            ax1.set_ylabel('Absorbance [a.u.]', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_sample_spectra(self, save_fig=True):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            # Convert sample spectra dict to a list for alternating plotting
            sample_dict = {stype: list(self.sample_spectra[stype]) for stype in self.sample_spectra.keys()}
            max_spectra = max(len(s) for s in sample_dict.values())

            alpha = min(1, 5/max_spectra)

            # Plot spectra alternatingly
            for i in range(max_spectra):
                for stype in sample_dict.keys():
                    if i < len(sample_dict[stype]):  # Check if datatype has a spectrum at index i
                        color = self.parent.color_gen.get_color(stype)
                        ax1.plot(x, sample_dict[stype][i], lw=0.5, alpha=alpha, color=color)
                        ax2.plot(x, sample_dict[stype][i], lw=0.5, alpha=alpha, color=color)

            # Plot the mean spectra on top for clarity
            for stype in self.sample_spectra.keys():
                color = self.parent.color_gen.get_color(stype)
                mean_spectrum = np.mean(self.sample_spectra[stype], axis=0)
                ax1.plot(x, mean_spectrum, label=stype, lw=0.5, color=color)
                ax2.plot(x, mean_spectrum, label=stype, lw=0.5, color=color)

            # Add vertical lines for reference
            if self.wavenumber_index_to_look_at != -9999:
                ax1.axvline(x=x[self.wavenumber_index_to_look_at], color='grey', linestyle='--', label='Wavenumber for distribution plots')
                ax2.axvline(x=x[self.wavenumber_index_to_look_at], color='grey', linestyle='--')

            # Add horizontal lines at y=0
            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)

            title = f'{sum(len(v) for v in sample_dict.values())} sample spectra {self.parent.title_addition}'
            ax1.set_title(title, fontsize=14)
            ax1.set_ylabel('Absorbance [a.u.]', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Add broken axis diagonal lines
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75), 1+d*(20/75)), (-d, +d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75), 1+d*(20/75)), (1-d, 1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d, d), (-d, +d), linewidth=1, **kwargs)
            ax2.plot((-d, d), (1-d, 1+d), linewidth=1, **kwargs)

            # Hide unnecessary spines
            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.tight_layout()
            
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")

            plt.show()


        def plot_differential_fingerprint(self, save_fig=True):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            
            for stype in self.diff_fp.keys():
                ax1.plot(x, self.diff_fp[stype], label=f'diff fp {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, self.diff_fp[stype], color=self.parent.color_gen.get_color(stype))
                ax1.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.1, label=f'std diff fp {stype[0]}-{stype[1]}')
                ax2.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.1)
                ax1.plot(x, self.diff_fp[stype] - self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                ax2.plot(x, self.diff_fp[stype] - self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                ax1.plot(x, self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                ax2.plot(x, self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), linewidth=0.5)

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)
            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            title = f'Differential fingerprints {self.parent.title_addition}'
            ax1.set_title(title, fontsize=14)
            ax1.set_ylabel('Difference in Mean', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_effect_size(self, save_fig=True):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            for stype in self.effect_size.keys():
                ax1.plot(x, self.effect_size[stype], label=f'effect size {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, self.effect_size[stype], color=self.parent.color_gen.get_color(stype))

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)
            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            title = f'Effect sizes {self.parent.title_addition}'
            ax1.set_title(title, fontsize=14)
            ax1.set_ylabel('Effect size', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            temp_mins = [min(self.effect_size[stype]) for stype in self.effect_size.keys()]
            temp_maxs = [max(self.effect_size[stype]) for stype in self.effect_size.keys()]
            ax1.set_ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])
            ax2.set_ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()

        def plot_ttest(self, save_fig=True):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
            ymins = []
            ymaxs = []

            for stype in self.pvalues.keys():
                ydata = self.pvalues[stype]
                ax1.plot(x, ydata, label=f'p-value {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, ydata, label=f'p-value {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype))
                ymins.append(min(ydata))
                ymaxs.append(max(ydata))

            # Broken axis setup
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)
            ax1.axhline(y=0.05, color='grey', linestyle='--', label='p = 0.05')
            ax2.axhline(y=0.05, color='grey', linestyle='--', label='p = 0.05')

            # Log scale
            ax1.set_yscale('log')
            ax2.set_yscale('log')

            # Diagonal lines
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            # Axis limits
            ymin = min(1e-5, min(ymins) * 0.9)
            ymax = max(1, max(ymaxs) * 1.1)
            ax1.set_ylim([ymin, 1.05])
            ax2.set_ylim([ymin, 1.05])

            # Labels and title
            ax1.set_ylabel('p-value', fontsize=12)
            ax1.set_xlabel('Wavenumber', fontsize=12)
            title = 'Two-tailed p-value of Student t-test'
            ax1.set_title(title)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax2.legend(loc='lower right')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


    class Metric(Evaluator.Metric):
        def __init__(self, parent):
            super().__init__(parent)


    class Loss(Evaluator.Loss):
        def __init__(self, parent):
            super().__init__(parent)


    class Condition(Evaluator.Condition):
        def __init__(self, parent):
            super().__init__(parent)


        def plot_differential_fingerprint(self, std_to_plot=0, save_fig=True):
            x = self.parent.vec
            figs = []

            for label in self.diff_fp_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
                for stype in self.diff_fp.keys():
                    ax1.plot(x, self.diff_fp[stype][label], label=f'{stype}', color=self.parent.color_gen.get_color(stype))
                    ax2.plot(x, self.diff_fp[stype][label], color=self.parent.color_gen.get_color(stype))

                    ax1.fill_between(
                        x,
                        self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                        self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                        color=self.parent.color_gen.get_color(stype), alpha=0.1, label=f'std {stype}')
                
                    ax2.fill_between(
                        x,
                        self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                        self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                        color=self.parent.color_gen.get_color(stype), alpha=0.1)
                    
                    ax1.plot(x, self.diff_fp[stype][label] + self.std_diff_fp[stype][label], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                    ax2.plot(x, self.diff_fp[stype][label] + self.std_diff_fp[stype][label], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                    ax1.plot(x, self.diff_fp[stype][label] - self.std_diff_fp[stype][label], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                    ax2.plot(x, self.diff_fp[stype][label] - self.std_diff_fp[stype][label], color=self.parent.color_gen.get_color(stype), linewidth=0.5)


                # Broken axis settings
                ax1.set_xlim(1000, 1800)
                ax2.set_xlim(2800, 3000)
                ax1.axhline(y=0, color='gray', linestyle='--')
                ax2.axhline(y=0, color='gray', linestyle='--')

                # Diagonal lines for broken axis
                d = .015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
                ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.set_major_locator(ticker.NullLocator())

                # Set axis labels and title
                ax1.set_xlabel("Feature Index", fontsize=12)
                ax1.set_ylabel("Difference in Mean", fontsize=12)
                title = f"Differential Fingerprints {label} {self.parent.title_addition}"
                ax1.set_title(title, fontsize=14)

                # Add legend
                ax1.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()


        def plot_effect_size(self, save_fig=True):
            x = self.parent.vec

            for label in self.effect_size_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
                ymins = []
                ymaxs = []
                for stype in self.effect_size.keys():
                    ax1.plot(x, self.effect_size[stype][label], label=f'{stype}', color=self.parent.color_gen.get_color(stype))
                    ax2.plot(x, self.effect_size[stype][label], label=f'{stype}', color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.effect_size[stype][label]))
                    ymaxs.append(max(self.effect_size[stype][label]))

                # Broken axis settings
                ax1.set_xlim(1000, 1800)
                ax2.set_xlim(2800, 3000)
                ax1.axhline(y=0, color='gray', linestyle='--')
                ax2.axhline(y=0, color='gray', linestyle='--')

                # Diagonal lines for broken axis
                d = .015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
                ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.set_major_locator(ticker.NullLocator())

                # Set axis labels and title
                ax1.set_xlabel("Wavenumber", fontsize=12)
                ax1.set_ylabel("Effect size", fontsize=12)
                title = f"Effect sizes {label} {self.parent.title_addition}"
                ax1.set_title(title, fontsize=14)

                # Set y-axis limits based on data range
                ax1.set_ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                ax2.set_ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])

                # Add legend
                ax1.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()



        def plot_correlation(self, save_fig=True):
            x = self.parent.vec

            for label in self.correlation_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
                ymins = []
                ymaxs = []
                for stype in self.correlation.keys():
                    ax1.plot(x, self.correlation[stype][label], label=f'{stype}', color=self.parent.color_gen.get_color(stype))
                    ax2.plot(x, self.correlation[stype][label], label=f'{stype}', color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.correlation[stype][label]))
                    ymaxs.append(max(self.correlation[stype][label]))

                # Broken axis settings
                ax1.set_xlim(1000, 1800)
                ax2.set_xlim(2800, 3000)
                ax1.axhline(y=0, color='gray', linestyle='--')
                ax2.axhline(y=0, color='gray', linestyle='--')

                # Diagonal lines for broken axis
                d = .015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
                ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.set_major_locator(ticker.NullLocator())

                # Set axis labels and title
                ax1.set_xlabel("Wavenumber", fontsize=12)
                ax1.set_ylabel("Correlation", fontsize=12)
                title = f"Correlation {label} {self.parent.title_addition}"
                ax1.set_title(title, fontsize=14)

                # Set y-axis limits based on data range
                ax1.set_ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                ax2.set_ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])

                # Add legend
                ax1.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()


        def plot_ttest(self, save_fig=True):
            x = self.parent.vec

            for label in self.ttest_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
                ymins = []
                ymaxs = []

                for stype in self.pvalues.keys():
                    ydata = self.pvalues[stype][label]
                    ax1.plot(x, ydata, label=stype, color=self.parent.color_gen.get_color(stype))
                    ax2.plot(x, ydata, label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(ydata))
                    ymaxs.append(max(ydata))

                # Broken axis limits
                ax1.set_xlim(1000, 1800)
                ax2.set_xlim(2800, 3000)

                # Reference line at p = 0.05
                ax1.axhline(0.05, color='grey', linestyle='--', label='p = 0.05')
                ax2.axhline(0.05, color='grey', linestyle='--')

                # Log scale for y-axis
                ax1.set_yscale('log')
                ax2.set_yscale('log')

                # Diagonal lines to indicate break in axis
                d = .015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d*(20/75),1+d*(20/75)), (-d,+d), linewidth=1, **kwargs)
                ax1.plot((1-d*(20/75),1+d*(20/75)), (1-d,1+d), linewidth=1, **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d,d), (-d,+d), linewidth=1, **kwargs)
                ax2.plot((-d,d), (1-d,1+d), linewidth=1, **kwargs)

                # Hide axis spines and ticks where appropriate
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.set_major_locator(ticker.NullLocator())

                # Labels and title
                ax1.set_xlabel("Wavenumber", fontsize=12)
                ax1.set_ylabel("p-value", fontsize=12)
                title = f"P-values {label} {self.parent.title_addition}"
                ax1.set_title(title, fontsize=14)

                # Set y-axis limits
                ymin = min(1e-5, min(ymins)*0.9)
                ymax = max(1, max(ymaxs)*1.1)
                ax1.set_ylim([ymin, 1.05])
                ax2.set_ylim([ymin, 1.05])

                # Legend
                ax2.legend(loc='lower right', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()
