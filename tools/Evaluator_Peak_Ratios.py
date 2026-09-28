import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

from .calculation_functions import *
from .Evaluator import *
from .calculation_functions import _NoneScaler
from .Loss_Metric_Manager import *


class Evaluator_Peak_Ratios(Evaluator):
    '''
    Converts FTIR spectra into peak ratios in the __init__ function.
    Otherwise inherits all calculations (i.e. add_ functions) from Evaluator.
    Plots which would show spectra are now changed to bar plots.
    '''
    def __init__(self, spectra_map, scaler=None, loss_metric_manager=None, vec=None, init_figure_settings_once=False):
        super().__init__(
            spectra_map=spectra_map,
            scaler=scaler,
            loss_metric_manager=loss_metric_manager,
            vec=vec,
            init_figure_settings_once=init_figure_settings_once)
        self.spectra_scaled = OrderedDict(spectra_map.copy())
        self.loss_metric_manager = loss_metric_manager if loss_metric_manager is not None else Loss_Metric_Manager()

        if scaler is None:
            scaler = _NoneScaler()
        else: 
            self.scaler = scaler

        # Scale data
        self.spectra = {}
        self.default_split_masks = {}
        for stype in self.spectra_scaled.keys():
            self.spectra[stype] = self.scaler.inverse_transform(self.spectra_scaled[stype])
            self.default_split_masks[stype] = {f"mask_0": np.array([True]* len(self.spectra[stype]))}
            
        self.vec = vec if vec is not None else np.linspace(0, self.spectra[stype].shape[1] - 1, self.spectra[stype].shape[1])

        for i, stype in enumerate(self.spectra.keys()):
            if i == len(list(self.spectra.keys()))-1:
                self.spectra[stype], self.vec = peak_ratio_numpy(self.spectra[stype], self.vec)
            else:
                self.spectra[stype], _ = peak_ratio_numpy(self.spectra[stype], self.vec)
            self.scaler = StandardScaler()
            self.spectra_scaled[stype] = self.scaler.fit_transform(self.spectra[stype])

        # for the paper
        self.vec = np.linspace(0, self.spectra[stype].shape[1] - 1, self.spectra[stype].shape[1], dtype=int)

        # Subclasses can be used without ()
        self.dist = self.Dist(parent=self)
        self.metric = self.Metric(parent=self)
        self.loss = self.Loss(parent=self)
        self.condition = self.Condition(parent=self)

        # Make figure settings once (default setup)
        if not Evaluator_Peak_Ratios._figure_settings_initialized and init_figure_settings_once:
            self.init_figure_settings()
            Evaluator_Peak_Ratios._figure_settings_initialized = True
        else:
            self.init_figure_settings()


    class Dist(Evaluator.Dist):
        def __init__(self, parent):
            super().__init__(parent)
            self._add_box_plots = False
            self.wavenumber_index_to_look_at = -9999

        def add_box_plots(self, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.box_plot_data = {}
            for stype in stypes:
                self.box_plot_data[stype] = self.parent.spectra[stype]

            self._add_box_plots = True


        def plot_sample_spectrum(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(7, 4))

            for stype in self.sample_spectrum.keys():
                plt.plot(x, self.sample_spectrum[stype], label=f'{stype}', marker='.', linestyle='', lw=0.5, color=self.parent.color_gen.get_color(stype))
            
            title = f'Sample peak ratios {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.legend(loc='upper left', fontsize=10)
            plt.xlabel('Peak ratio index', fontsize=12)
            #plt.xticks(rotation=90)
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_sample_spectra(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(7, 4))
            n_samples = len(list(self.sample_spectra.values()))

            for stype in self.sample_spectra.keys():
                plt.plot(x, np.mean(self.sample_spectra[stype], axis=0), label=stype, lw=0.5, marker='.', linestyle='', color=self.parent.color_gen.get_color(stype), alpha=0.5)
                for i in range(len(self.sample_spectra[stype])):
                    plt.plot(x, self.sample_spectra[stype][i], lw=0.5, marker='.', linestyle='', alpha=1/n_samples, color=self.parent.color_gen.get_color(stype))
            if self.wavenumber_index_to_look_at != -9999:
                plt.axvline(x=x[self.wavenumber_index_to_look_at], label='peak ratio for distribution plots', color='grey', ls='dashed')

            title = f'{len(self.sample_spectra[stype])} sample peak ratios {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.legend(loc='upper left', fontsize=10)
            plt.xlabel('Peak ratio index', fontsize=12)
            #plt.xticks(rotation=90)
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        # def plot_box_plots(self, save_fig=True):
        #     x = self.parent.vec
        #     for stype in self.box_plot_data.keys():
        #         fig = plt.figure(figsize=(8, 4.8))
        #         plt.boxplot(self.box_plot_data[stype], tick_labels=x, showfliers=False)
        #         title = f'Boxplots {stype} {self.parent.title_addition}'
        #         plt.title(title)
        #         plt.xticks(rotation=90)
        #         if save_fig:
        #             self.parent.plt_save_fig(fig, title, dir="Dist")
        #         plt.show()

        def plot_box_plots(self, save_fig=True):
            x = self.parent.vec
            fig, ax = plt.subplots(figsize=(7, 4))
            
            positions = range(1, len(x) + 1)  # x-axis positions
            width = 0.2  # Adjust width for better visibility
            legend_handles = []  # Store legend handles
            
            for i, (stype, data) in enumerate(self.box_plot_data.items()):
                pos_offset = [p + (i - len(self.box_plot_data) / 2) * width for p in positions]  # Offset for each boxplot
                ax.boxplot(data, positions=pos_offset, widths=width, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=self.parent.color_gen.get_color(stype), alpha=0.6))  # Use colors for differentiation
                legend_handles.append(plt.Line2D([0], [0], color=self.parent.color_gen.get_color(stype), alpha=0.6, lw=1, label=stype))  # Create legend handle
            ax.legend(handles=legend_handles, loc='upper right', fontsize=10)  # Add legend in the top right
            title = f'Boxplots {self.parent.title_addition}'
            ax.set_title(title, fontsize=14)
            ax.set_ylabel('Ratio of absorbances', fontsize=12)
            ax.set_xlabel('Peak ratio index', fontsize=12)
            ax.set_xticks(positions[::5])
            ax.set_xticklabels(x[::5])
            
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            
            plt.show()


        def plot_differential_fingerprint(self, save_fig=True):
            x = self.parent.vec
            x_labels = self.parent.vec  # List of names (strings)
            x_positions = np.arange(len(x_labels))  # Numeric positions for bars

            fig = plt.figure(figsize=(7, 4))

            num_stypes = len(self.diff_fp.keys())
            bar_width = 0.8 / num_stypes  # Make sure bars fit without overlapping
            offsets = np.linspace(-bar_width * (num_stypes - 1) / 2, 
                                bar_width * (num_stypes - 1) / 2, num_stypes)  # Center bars

            for i, (stype, offset) in enumerate(zip(self.diff_fp.keys(), offsets)):
                bar_color = self.parent.color_gen.get_color(stype)

                plt.bar(x_positions + offset, self.diff_fp[stype], width=bar_width, label=f'diff fp {stype[0]}-{stype[1]}', 
                        color=bar_color, alpha=0.6, edgecolor=bar_color, linewidth=1)

                plt.errorbar(x_positions + offset, self.diff_fp[stype], yerr=self.std_diff_fp[stype], fmt='none', 
                            color=bar_color, elinewidth=1, capsize=2, capthick=1, 
                            label=f'std diff fp {stype[0]}-{stype[1]}')
                
            # for stype in self.diff_fp.keys():
            #     plt.plot(x, self.diff_fp[stype], label=f'diff fp {stype}', color=self.parent.color_gen.get_color(stype))
            #     plt.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std diff fp {stype}')
            #     plt.plot(x, self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
            #     plt.plot(x, self.diff_fp[stype] - self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), linewidth=0.5)

            plt.xlabel('Peak ratio index', fontsize=12)
            plt.ylabel('Difference in Mean', fontsize=12)
            title = f'Differential fingerprints {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.legend(loc='upper left', fontsize=10)
            #plt.xticks(x_positions, x_labels, rotation=90)  

            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_effect_size(self, save_fig=True):
            x = self.parent.vec
            x_labels = self.parent.vec  # List of peak ratio labels
            x_positions = np.arange(len(x_labels))  # Numeric positions for bars

            fig = plt.figure(figsize=(7, 4))

            num_stypes = len(self.effect_size.keys())
            bar_width = 0.8 / num_stypes  # Make sure bars fit without overlapping
            offsets = np.linspace(-bar_width * (num_stypes - 1) / 2, 
                                bar_width * (num_stypes - 1) / 2, num_stypes)  # Center bars

            for i, (stype, offset) in enumerate(zip(self.effect_size.keys(), offsets)):
                bar_color = self.parent.color_gen.get_color(stype)
                plt.bar(x_positions + offset, self.effect_size[stype], width=bar_width, label=f'effect size {stype[0]}-{stype[1]}', 
                        color=bar_color, alpha=0.6, edgecolor=bar_color, linewidth=1)

            plt.xlabel('Peak ratio index', fontsize=12)
            plt.ylabel('Effect size', fontsize=12)

            # Set dynamic y-limits
            temp_mins = [min(self.effect_size[stype]) for stype in self.effect_size.keys()]
            temp_maxs = [max(self.effect_size[stype]) for stype in self.effect_size.keys()]
            plt.ylim([min(-1, min(temp_mins) * 1.05), max(1, max(temp_maxs) * 1.05)])

            title = f'Effect sizes {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.legend(loc='upper left', fontsize=10)

            # Ensure x-axis labels are correct
            #plt.xticks(x_positions, x_labels, rotation=90)

            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()

        def plot_ttest(self, save_fig=True):
            x_labels = self.parent.vec  # List of peak ratio labels
            x_positions = np.arange(len(x_labels))  # Numeric positions for bars

            fig = plt.figure(figsize=(7, 4))

            num_stypes = len(self.pvalues.keys())
            bar_width = 0.8 / num_stypes  # Make sure bars fit without overlapping
            offsets = np.linspace(-bar_width * (num_stypes - 1) / 2, 
                                bar_width * (num_stypes - 1) / 2, num_stypes)  # Center bars
            
            for i, (stype, offset) in enumerate(zip(self.pvalues.keys(), offsets)):
                bar_color = self.parent.color_gen.get_color(stype)
                plt.bar(x_positions + offset, self.pvalues[stype], width=bar_width, label=f'effect size {stype[0]}-{stype[1]}', 
                        color=bar_color, alpha=0.6, edgecolor=bar_color, linewidth=1)

            plt.hlines(0.05, x_labels[0], x_labels[-1], color='grey', linestyle='--', label='p = 0.05')

            # Labels and title
            plt.ylabel('p-value')
            plt.yscale('log')
            plt.xlabel('Peak ratio index')
            title = 'Two-tailed p-value of Student t-test'
            plt.title(title)

            # Show legend and plot
            plt.legend(loc='lower right')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()



        def plot_sample_distribution(self, bin_frequency=100, hist_stds_to_include = 3, focus_stype=0, save_fig=True):
            if focus_stype == 0: focus_stype=list(self.hist.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.hist.keys())[focus_stype]
            x = self.parent.vec
            hist_range_max = np.mean(self.hist[focus_stype]) + hist_stds_to_include * np.std(self.hist[focus_stype].astype(float))
            hist_range_min = np.mean(self.hist[focus_stype]) - hist_stds_to_include * np.std(self.hist[focus_stype].astype(float))
            
            fig = plt.figure(figsize=(7, 4))
            for stype in self.hist.keys():
                plt.hist(
                    self.hist[stype][np.logical_and(self.hist[stype] > hist_range_min, self.hist[stype] < hist_range_max)],
                    bins=bin_frequency, color=self.parent.color_gen.get_color(stype), alpha=0.5, density=True)
                x_values = np.linspace(hist_range_min, hist_range_max, 1000)
                plt.plot(x_values, self.hist_kde[stype](x_values), color=self.parent.color_gen.get_color(stype), label=stype, linestyle='-')
            plt.xlim([hist_range_min, hist_range_max])
            plt.legend(loc='upper left', fontsize=10)
            title = f'Distribution at peak ratio {self.parent.vec[self.wavenumber_index_to_look_at]} {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.xlabel('L2 Norm', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            if save_fig:
                self.parent.plt_save_fig(fig, 'Sample Distribution', dir="Dist")
            plt.show()


        def plot_kde(self, stds_to_include = 2, focus_stype=None, save_fig=True):
            if type(focus_stype) == int: focus_stype=list(self.density.keys())[focus_stype]
            elif type(focus_stype) == str: focus_stype=focus_stype

            x = self.parent.vec
            ymaxs = []
            for stype in self.density.keys():
                for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                    ymaxs.append(max(self.density[stype][frequency_index](self.x_range_kde))*1.25)

            for stype in self.density.keys():
                fig = plt.figure(figsize=(7, 4))
                for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                    plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), color=self.parent.color_gen.get_color(stype), alpha=0.2)
                plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), alpha=0.3, label=stype, color=self.parent.color_gen.get_color(stype))
                plt.legend(fontsize=10)
                title = f'Density distributions of all peak ratios {stype} {self.parent.title_addition}'
                plt.title(title, fontsize=14)
                plt.xlim([-1*stds_to_include, stds_to_include])
                if focus_stype is not None:
                    plt.ylim([0, max(self.density[focus_stype][frequency_index](self.x_range_kde))*1.25])
                else:
                    plt.ylim([0, max(max(ymaxs), 0.55)])
                plt.xlabel('Standard deviations', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Dist")
                plt.show()


        def plot(self, save_figs=True):
            if self._add_sample_spectrum:
                self.plot_sample_spectrum(save_fig=save_figs)
            if self._add_sample_spectra:
                self.plot_sample_spectra(save_fig=save_figs)
            if self._add_differential_fingerprint:
                self.plot_differential_fingerprint(save_fig=save_figs)
            if self._add_effect_size:
                self.plot_effect_size(save_fig=save_figs)
            if self._add_sample_distribution:
                self.plot_sample_distribution(save_fig=save_figs)
            if self._add_kde:
                self.plot_kde(save_fig=save_figs)
            if self._add_pca:
                self.plot_pca(save_fig=save_figs)
            if self._add_pca_kde:
                self.plot_pca_kde(save_fig=save_figs)
            if self._add_correlation_matrix:
                self.plot_correlation_matrix(save_fig=save_figs)
            if self._add_diff_correlation_matrix:
                self.plot_diff_correlation_matrix(save_fig=save_figs)
            if self._add_vector_length:
                self.plot_vector_length(save_fig=save_figs)
            if self._add_box_plots:
                self.plot_box_plots(save_fig=save_figs)
            if self._add_ttest:
                self.plot_ttest(save_fig=save_figs)


    class Metric(Evaluator.Metric):
        def __init__(self, parent):
            super().__init__(parent)


    class Loss(Evaluator.Loss):
        def __init__(self, parent):
            super().__init__(parent)


    class Condition(Evaluator.Condition):
        def __init__(self, parent):
            super().__init__(parent)


        def plot_differential_fingerprint(self, save_fig=True):
            
            x_labels = self.parent.vec
            x_positions = np.arange(len(x_labels))
            figs = []

            for label in self.diff_fp_labels:
                x = self.parent.vec
                fig = plt.figure(figsize=(7, 4))
                num_stypes = len(self.diff_fp.keys())
                bar_width = 0.8 / num_stypes  
                offsets = np.linspace(-bar_width * (num_stypes - 1) / 2, 
                                    bar_width * (num_stypes - 1) / 2, num_stypes)  

                for stype, offset in zip(self.diff_fp.keys(), offsets):
                    bar_color = self.parent.color_gen.get_color(stype)

                    plt.bar(x_positions + offset, self.diff_fp[stype][label], width=bar_width, 
                            label=f'diff fp {stype}', color=bar_color, alpha=0.6, 
                            edgecolor=bar_color, linewidth=1)

                    plt.errorbar(x_positions + offset, self.diff_fp[stype][label], 
                                yerr=self.std_diff_fp[stype][label], fmt='none', label=f'std diff fp {stype}',
                                color=bar_color, elinewidth=1, capsize=2, capthick=1)

                    # plt.plot(x, self.diff_fp[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))
                    # plt.fill_between(
                    #     x,
                    #     self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                    #     self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                    #     color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std {stype}')
                    # plt.plot(x, self.diff_fp[stype][label] + self.std_diff_fp[stype][label], color=self.parent.color_gen.get_color(stype), linewidth=0.5)
                    # plt.plot(x, self.diff_fp[stype][label] - self.std_diff_fp[stype][label], color=self.parent.color_gen.get_color(stype), linewidth=0.5)

                plt.xlabel("Peak Ratio index", fontsize=12)
                plt.ylabel("Difference in Mean", fontsize=12)
                title = f"Differential Fingerprints {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.legend(loc='upper left', fontsize=10)
                #plt.xticks(x_positions, x_labels, rotation=90)
                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()


        def plot_effect_size(self, save_fig=True):

            x_labels = self.parent.vec
            x_positions = np.arange(len(x_labels))

            for label in self.effect_size_labels:
                x = self.parent.vec
                fig = plt.figure(figsize=(7, 4))
                num_stypes = len(self.effect_size.keys())
                bar_width = 0.8 / num_stypes  
                offsets = np.linspace(-bar_width * (num_stypes - 1) / 2, 
                                    bar_width * (num_stypes - 1) / 2, num_stypes)  

                ymins, ymaxs = [], []

                for stype, offset in zip(self.effect_size.keys(), offsets):

                    bar_color = self.parent.color_gen.get_color(stype)
                    plt.bar(x_positions + offset, self.effect_size[stype][label], width=bar_width, 
                            label=stype, color=bar_color, alpha=0.6, edgecolor=bar_color, linewidth=1)

                    # plt.plot(x, self.effect_size[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.effect_size[stype][label]))
                    ymaxs.append(max(self.effect_size[stype][label]))

                plt.xlabel("Peak Ratio index", fontsize=12)
                plt.ylabel("Effect Size", fontsize=12)
                title = f"Effect Size {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.ylim([min(-1, min(ymins) * 1.05), max(1, max(ymaxs) * 1.05)])
                plt.legend(loc='upper left', fontsize=10)
                #plt.xticks(x_positions, x_labels, rotation=90)
                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()

        
        def plot_ttest(self, save_fig=True):
            x_labels = self.parent.vec
            x_positions = np.arange(len(x_labels))

            for label in self.ttest_labels:
                fig = plt.figure(figsize=(7, 4))
                num_stypes = len(self.pvalues.keys())
                bar_width = 0.8 / num_stypes  
                offsets = np.linspace(-bar_width * (num_stypes - 1) / 2, 
                                    bar_width * (num_stypes - 1) / 2, num_stypes)  

                for stype, offset in zip(self.pvalues.keys(), offsets):

                    bar_color = self.parent.color_gen.get_color(stype)
                    plt.bar(x_positions + offset, self.pvalues[stype][label], width=bar_width, 
                            label=stype, color=bar_color, alpha=0.6, edgecolor=bar_color, linewidth=1)

                plt.hlines(0.05, x_labels[0], x_labels[-1], color='grey', linestyle='--', label='p = 0.05')

                plt.xlabel("Peak ratio index", fontsize=12)
                plt.ylabel("p-value", fontsize=12)
                plt.yscale('log')
                title = f"P-values {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.legend(loc='lower right', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()