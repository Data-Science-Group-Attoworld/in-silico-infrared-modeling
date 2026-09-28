# Import packages
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

from itertools import combinations, product
from collections import defaultdict

from hotelling.stats import hotelling_t2
import scipy
from scipy.stats import gaussian_kde

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# import other modules
from .ColorGenerator import *
from .calculation_functions import *
from .calculation_functions import _NoneScaler
from .Loss_Metric_Manager import *


class Evaluator:
    '''
    Class which incorporates standard calculations and plots. 
    Serves as a parent class to Evaluator_FTIR and Evaluator_Peak_Ratios.
    Tip for easy navigation when using vs code:
        press ctrl+k then ctrl+0 to collapse all methods
        press ctrl+k then ctrl+j to unfold all
    '''
    # Class-level attribute to track if figure settings have been initialized
    _figure_settings_initialized = True

    def __init__(self, spectra_map, scaler=None, loss_metric_manager=None, vec=None, init_figure_settings_once=False):
        """
        Initializes the Evaluator base class

        spectra_map: dict: 
            keys = datatype as string (e.g. "real", "simulated")
            values = dataset as np.ndarray
        scaler: fitted scikit-learn scaler or custom scaler with similar format (i.e. callable functions named transform andinverse_transform)
        directory: str: directory to store the figures in
        loss_metric_manager: loss metric manager class object: Stores metrics and losses when using Evaluator inside a loop (e.g. when training a NN)
        vec: np.array or list: x-axis values for the spectra (e.g. times for FTIR data)
        init_once: boolean: if True: initializes the figure settings only once (usefull when using Evaluator inside a loop, otherwise plot directories will be overwritten in each iteration in the loop) 
        """

        self.spectra_scaled = spectra_map.copy()
        self.loss_metric_manager = loss_metric_manager if loss_metric_manager is not None else Loss_Metric_Manager()

        if scaler is None:
            self.scaler = _NoneScaler()
        else: 
            self.scaler = scaler

        # Scale data
        self.spectra = {}
        for stype in self.spectra_scaled.keys():
            self.spectra[stype] = self.scaler.inverse_transform(self.spectra_scaled[stype])

        # set x-axis values for plots
        self.vec = vec if vec is not None else np.linspace(0, self.spectra[stype].shape[1] - 1, self.spectra[stype].shape[1])

        # Subclasses can be used without ()
        self.dist = self.Dist(parent=self)
        self.metric = self.Metric(parent=self)
        self.loss = self.Loss(parent=self)
        self.condition = self.Condition(parent=self)

        # Make figure settings once (default setup)
        if not Evaluator._figure_settings_initialized and init_figure_settings_once:
            self.init_figure_settings()
            Evaluator._figure_settings_initialized = True
        else:
            self.init_figure_settings()


    def init_figure_settings(self):
            self.color_gen = ColorGenerator(scale='matplotlib', colors='default', combination_hierarchy='same as stypes')
            self.title_addition = ''
            self.figure_directory = './Results'
            self.figure_filetypes =['pdf']


    def handle_stpye_kwargs(self, kwargs):
        result = {}

        if 'stypes' in kwargs:
            stypes = kwargs['stypes']
            if stypes == 'all':
                result['stypes'] = list(self.spectra.keys())
            elif isinstance(stypes, list):
                result['stypes'] = stypes
            else:
                raise ValueError("Invalid type for 'stypes'. Expected 'all' or a list of valid stypes.")
            
        elif 'stype_combinations' in kwargs:
            stype_combinations = kwargs['stype_combinations']
            if stype_combinations == 'all':
                result['stype_combinations'] = list(combinations(self.spectra.keys(), 2))
            elif isinstance(stype_combinations, list):
                result['stype_combinations'] = stype_combinations
            else:
                raise ValueError("Invalid type for 'stype_combinations'. Expected 'all' or a list of tupels containing the combinations of stypes")

            result['stypes'] = list({element for tuple_ in result['stype_combinations'] for element in tuple_})
        
        elif 'stype_products' in kwargs:
            stype_products = kwargs['stype_products']
            if stype_products == 'all':
                result['stype_products'] = list(product(self.spectra.keys(), repeat=2))
            elif isinstance(stype_products, list):
                result['stype_products'] = stype_products
            else:
                raise ValueError("Invalid type for 'stype_products'. Expected 'all' or a list of tupels containing the combinations of stypes")
            
            result['stypes'] = list({element for tuple_ in result['stype_products'] for element in tuple_})

        else:   # default to 'all'
            result['stype_combinations'] = list(combinations(self.spectra.keys(), 2))
            result['stype_products'] = list(product(self.spectra.keys(), repeat=2))
            result['stypes'] = list(self.spectra.keys())

        return result
    

    def figure_settings(self, title_addition='' ,directory='./Results', filetypes=['pdf'], _reset=True, colors='default', combination_hierarchy='same as stypes', skip_directory_check=False):
        """
        Configure figure settings with optional parameters.
        
        Args:
            title_addition: str: Additional text to be displayed after the title of each figure (E.g. useful for the epoch during training)
            directory: Same as the directory in the __init__
            filetypes: list of strings: specify the filetype to store the plots (E.g. 'png' or 'pdf'). Can be anything which is supported by matplotlib

        """
        self.color_gen = ColorGenerator(scale='matplotlib', colors=colors, combination_hierarchy=combination_hierarchy)
        self.title_addition = title_addition

        # Create or verify the directory
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        else:
            # Notify and ask the user for confirmation if _resetting
            if _reset:
                if skip_directory_check == False:
                    print(f"Directory '{directory}' already exists.")
                    print("If you proceed, all data in this directory will be deleted. Otherwise, the figures will be plotted but not saved")
                    proceed = input("Do you want to proceed? (y/n): ").strip().lower()
                else:
                    proceed = 'y'
                if proceed == 'y':
                    # Delete all data in the directory
                    for item in os.listdir(directory):
                        item_path = os.path.join(directory, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)  # Remove files or symbolic links
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)  # Remove directories
                    print(f"All data in '{directory}' has been deleted. It will be replaced by the plots below.")
                else:
                    print("Continuing with plotting without saving.")
                    return  # Exit the function if the user doesn't want to proceed

        # Update instance attributes for figure settings
        self.figure_directory = directory
        self.figure_filetypes = filetypes


    def plt_save_fig(self, fig, fig_name, dir):
        """
        Global helper function: Creates the directory for the plot if it doesn't exist and stores the plot in all filetypes specified.
        
        fig: matplotlib fig object
        fig_name: filename for the figure (usually the title)
        dir: subdir to self.figure_directory
        """
        if not os.path.isdir(f"{self.figure_directory}/{dir}"):
            os.makedirs(f"{self.figure_directory}/{dir}", exist_ok=True)
        for filetype in self.figure_filetypes:
            fig.savefig(f"{self.figure_directory}/{dir}/{fig_name}.{filetype}", dpi=300)

    # ----------------- #
    #   SUBCLASS DIST   #
    # ----------------- #

    class Dist():
        """
        Dist subclass for calculating and plotting all evaluations related to computations on the distributions
        The structure for this and all subsequent subclasses will be as follows:
        dist.add_{...} computes the relevant variables
        dist.plot_{...} creates a plot for the corresponding variables

        The dist subclass has add_ and plot_ functions for:
            sample_spectrum
            sample_spectra
            sample_distribution
            effect_size
            differential_fingerprint
            kde
            pca
            pca_kde (kde on the pca)
            correlation_matrix
            diff_correlation_matrix
            vector_length
        
        """
        def __init__(self, parent):
            self.parent = parent

            # Booleans for checking whether to automatically create a plot when calling .plot() 
            self._add_sample_spectrum = False
            self._add_sample_spectra = False
            self._add_sample_distribution = False
            self._add_effect_size = False
            self._add_differential_fingerprint = False
            self._add_kde = False
            self._add_pca = False
            self._add_pca_kde = False
            self._add_correlation_matrix = False
            self._add_diff_correlation_matrix = False
            self._add_vector_length = False
            self._add_ttest = False

            self.time_index_to_look_at = -9999 # if no time index is specified using add_sample_dist, the vertical line will be plottet outside the xlim

        # ----------------------- #
        #  Calculation functions  #
        # ----------------------- #

        def add_sample_spectrum(self, **kwargs):
            # selects the first spectrum of each stype (trivial function but keeps a uniform structure)
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.sample_spectrum = {}
            for stype in stypes:
                self.sample_spectrum[stype] = self.parent.spectra[stype][0]

            self._add_sample_spectrum = True


        def add_sample_spectra(self, n_spectra=100, **kwargs):
            # selects the first n_spectra spectra of each stype (trivial function but keeps a uniform sturcture)
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.sample_spectra = {}
            for stype in stypes:
                self.sample_spectra[stype] = self.parent.spectra[stype][:n_spectra]

            self._add_sample_spectra = True


        def add_sample_distribution(self, time_index_to_look_at=190, kde_spectra_cap=1000, **kwargs):
            # selects all y axis values at the specified time_index_to_look_at and calculates the kde of the resulting distribution
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.time_index_to_look_at = time_index_to_look_at
            temp_lens = [len(self.parent.spectra[stype]) for stype in stypes]
            equal_data_length = min(temp_lens)

            self.hist = {}
            self.hist_kde = {}
            for stype in stypes:
                self.hist[stype] = self.parent.spectra[stype][:equal_data_length, time_index_to_look_at]
                self.hist_kde[stype] = gaussian_kde(self.hist[stype][:kde_spectra_cap])
            
            self._add_sample_distribution = True


        def add_differential_fingerprint(self, **kwargs):
            # Calucaltes the differential fingerprint for each stpye combination
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            self.diff_fp = {}
            self.std_diff_fp = {}

            for stype in stype_combinations:
                mean0 = np.mean(self.parent.spectra[stype[0]], axis=0)
                mean1 = np.mean(self.parent.spectra[stype[1]], axis=0)
                var0 = np.var(self.parent.spectra[stype[0]], axis=0)
                self.diff_fp[stype] = np.array(mean0 - mean1, dtype=float)
                self.std_diff_fp[stype] = np.array(var0**0.5, dtype=float)

            self._add_differential_fingerprint = True


        def add_effect_size(self, convention='cohens d', **kwargs):
            # Calculates the effect_size for each stype combination. Conventions can be 'cohens d' or 'standardized mean difference'.
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']
            
            if convention == 'standardized mean difference':
                if self._add_differential_fingerprint == False:
                    self.add_differential_fingerprint(stype_combinations=stype_combinations)

                self.effect_size = {}
                for stype in stype_combinations:
                    self.effect_size[stype] = self.diff_fp[stype]/self.std_diff_fp[stype]

            elif convention == 'cohens d':
                self.effect_size = {}
                for stype in stype_combinations:
                    self.effect_size[stype] = cohens_d(self.parent.spectra[stype[0]].astype(float), self.parent.spectra[stype[1]].astype(float))
            else:
                raise ValueError('either use standardized mean difference or cohens d as a convention')

            self._add_effect_size = True


        def add_kde(self, inv_frequency=10, kde_spectra_cap=500, **kwargs):
            """
            Calculates the kde for every inv_frequency grid_point (i.e. times, timesteps,...)

            inv_frequency: int: distance between gridpoints for which the kde is calculated
            kde_spectra_cap: Limit the amount of data used to calculate the kde (entire large datasets such as H4H datasets take a long time to compute)
            """
            
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.kde_inv_frequency = inv_frequency
            self.density = {}
            for stype in stypes:
                density = []
                for k in range(0, len(self.parent.vec), self.kde_inv_frequency):
                    density.append(gaussian_kde(self.parent.spectra_scaled[stype][:kde_spectra_cap, k]))
                self.density[stype] = density

            self.x_range_kde = np.linspace(-2, 2, 100)

            self._add_kde = True


        def add_pca(self, fit_sytpe="real", **kwargs):
            # Calculates the first two principal components for each stpye
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.pcs = {}
            pca = PCA(n_components=2)
            
            self.pcs[fit_sytpe] = pca.fit_transform(self.parent.spectra_scaled[fit_sytpe])
            self.pcs_var = pca.explained_variance_ratio_
            
            for stype in stypes:
                if stype != fit_sytpe:
                    self.pcs[stype] = pca.transform(self.parent.spectra_scaled[stype])

            self._add_pca = True


        def add_pca_kde(self, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            if self._add_pca == False:
                self.add_pca()
                self._add_pca = False

            self.pcs_kde = {}
            for stype in stypes:
                self.pcs_kde[stype] = gaussian_kde(self.pcs[stype].T)

            self._add_pca_kde = True
 

        def add_correlation_matrix(self, **kwargs):
            # Calculates the correlation matrix for each stpye
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.correlation_matrix = {}
            for stype in stypes:
                self.correlation_matrix[stype] = np.corrcoef(self.parent.spectra_scaled[stype].T)

            self._add_correlation_matrix = True


        def add_diff_correlation_matrix(self, **kwargs):
            # Calculates the difference between correlation matrices for each combination of stypes
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            if self._add_correlation_matrix == False:
                self.add_correlation_matrix()
                self._add_correlation_matrix = False

            self.diff_correlation_matrix = {}
            for stype in stype_combinations:
                self.diff_correlation_matrix[stype] = self.correlation_matrix[stype[0]] - self.correlation_matrix[stype[1]]

            self._add_diff_correlation_matrix = True


        def add_vector_length(self, kde_spectra_cap=1000, **kwargs):
            # Calculates the l2 norm of each spectrum of each stype and calculates the kde over the resulting distribution
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.l2_norm = {}
            self.l2_norm_kde = {}
            for stype in stypes:
                self.l2_norm[stype] = np.linalg.norm(self.parent.spectra[stype], axis=1)
                self.l2_norm_kde[stype] = gaussian_kde(self.l2_norm[stype][:kde_spectra_cap])

            self._add_vector_length = True

        def add_ttest(self, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            self.pvalues = {}
            for stype in stype_combinations:
                _, self.pvalues[stype] = ttest_ind(self.parent.spectra[stype[0]], self.parent.spectra[stype[1]], axis=0)

            self._add_ttest = True

        # ----------------------- #
        #     Plot functions      #
        # ----------------------- #

        def plot_sample_spectrum(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for stype in self.sample_spectrum.keys():
                plt.plot(x, self.sample_spectrum[stype], label=f'{stype}', lw=0.5, color=self.parent.color_gen.get_color(stype))

            title = f'One sample spectrum {self.parent.title_addition}'
            plt.title(title)
            plt.xlabel('time')
            plt.ylabel('Absorbance')
            plt.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_sample_spectra(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            
            # Convert sample spectra dict to a list for alternating plotting
            sample_dict = {stype: list(self.sample_spectra[stype]) for stype in self.sample_spectra.keys()}
            max_spectra = max(len(s) for s in sample_dict.values())

            # Plot spectra alternatingly
            for i in range(max_spectra):
                for stype in sample_dict.keys():
                    if i < len(sample_dict[stype]):  # Check if the datatype has a spectrum at index i
                        plt.plot(x, sample_dict[stype][i], lw=0.5, alpha=0.5, color=self.parent.color_gen.get_color(stype))

            if self.time_index_to_look_at != -9999:
                plt.axvline(x=x[self.time_index_to_look_at], label='time for distribution plots', color='grey', ls='dashed')
            
            title = f'{sum(len(v) for v in sample_dict.values())} sample spectra {self.parent.title_addition}'
            plt.title(title)
            plt.xlabel('time')
            plt.ylabel('Absorbance')
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            
            plt.show()


        def plot_differential_fingerprint(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))

            for stype in self.diff_fp.keys():
                plt.plot(x, self.diff_fp[stype], label=f'diff fp {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype))
                plt.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std diff fp {stype[0]}-{stype[1]}')
            
            plt.xlabel('time')
            plt.ylabel('Difference in Mean')
            title = f'Differential fingerprints {self.parent.title_addition}'
            plt.title(title)
            plt.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_effect_size(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for stype in self.effect_size.keys():
                plt.plot(x, self.effect_size[stype], label=f'effect size {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype))
            plt.xlabel('time')
            plt.ylabel('Effect size')
            temp_mins = [min(self.effect_size[stype]) for stype in self.effect_size.keys()]
            temp_maxs = [max(self.effect_size[stype]) for stype in self.effect_size.keys()]
            plt.ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])
            title = f'Effect sizes {self.parent.title_addition}'
            plt.title(title)
            plt.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_sample_distribution(self, bin_frequency=100, hist_stds_to_include = 3, focus_stype=0, save_fig=True):
            if focus_stype == 0: focus_stype=list(self.hist.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.hist.keys())[focus_stype]
            else: focus_stype = None

            fig = plt.figure(figsize=(8, 4))
            hist_range_max = np.mean(self.hist[focus_stype]) + hist_stds_to_include * np.std(self.hist[focus_stype].astype(float))
            hist_range_min = np.mean(self.hist[focus_stype]) - hist_stds_to_include * np.std(self.hist[focus_stype].astype(float))

            for stype in self.hist.keys():
                plt.hist(
                    self.hist[stype][np.logical_and(self.hist[stype] > hist_range_min, self.hist[stype] < hist_range_max)],
                    bins=bin_frequency, color=self.parent.color_gen.get_color(stype), alpha=0.2, density=True)
                
                x_values = np.linspace(hist_range_min, hist_range_max, 1000)
                plt.plot(x_values, self.hist_kde[stype](x_values), color=self.parent.color_gen.get_color(stype), label=stype, linestyle='-')

            plt.xlim([hist_range_min, hist_range_max])
            plt.ylabel('Density')
            plt.xlabel('Absorbance')
            plt.legend(loc='upper left')
            title = f'Distributions at time {int(self.parent.vec[self.time_index_to_look_at])} 1/cm {self.parent.title_addition}'
            plt.title(title)
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, 'Sample Distribution', dir="Dist")    # not using the title here, because 1/cm would try to create a new directory
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
                fig = plt.figure(figsize=(8, 4))
                for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                    plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), color=self.parent.color_gen.get_color(stype), alpha=0.2)
                plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), alpha=0.3, label=stype, color=self.parent.color_gen.get_color(stype))
                plt.legend(loc='upper left')
                plt.xlim([-1*stds_to_include, stds_to_include])
                if focus_stype is not None:
                    plt.ylim([0, max(self.density[focus_stype][frequency_index](self.x_range_kde))*1.25])
                else:
                    plt.ylim([0, max(max(ymaxs), 0.55)])
                
                title = f'Density distributions of all times {stype} {self.parent.title_addition}'
                plt.title(title)
                plt.xlabel('Standard deviations')
                plt.ylabel('Density')
                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Dist")
                plt.show()


        def plot_kde_all_in_one(self, stds_to_include = 2, focus_stype=None, save_fig=True):
            if type(focus_stype) == int: focus_stype=list(self.density.keys())[focus_stype]
            elif type(focus_stype) == str: focus_stype=focus_stype

            x = self.parent.vec

            ymaxs = []
            for stype in self.density.keys():
                for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                    ymaxs.append(max(self.density[stype][frequency_index](self.x_range_kde))*1.25)

            fig = plt.figure(figsize=(8, 4))
            for stype in self.density.keys():
                for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                    plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), color=self.parent.color_gen.get_color(stype), alpha=0.2)
                plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), alpha=0.3, label=stype, color=self.parent.color_gen.get_color(stype))
            plt.legend(loc='upper left')
            plt.xlim([-1*stds_to_include, stds_to_include])
            if focus_stype is not None:
                plt.ylim([0, max(self.density[focus_stype][frequency_index](self.x_range_kde))*1.25])
            else:
                plt.ylim([0, max(max(ymaxs), 0.55)])

            title = f'Density distributions of all times {self.parent.title_addition}'
            plt.title(title)
            plt.xlabel('Standard deviations')
            plt.ylabel('Density')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_pca(self, n=100, outlier_threshold=3.0, save_fig=True):
            def remove_outliers(data, threshold=3.0):
                data = data.astype(float)
                mean = np.mean(data, axis=0)
                std_dev = np.std(data, axis=0)
                mask = np.all(np.abs((data - mean) / std_dev) <= threshold, axis=1)
                return data[mask]

            fig = plt.figure(figsize=(6, 6))
            legend_handles = [] 

            for stype in self.pcs.keys():
                # Filter out outliers
                filtered_pcs = remove_outliers(self.pcs[stype], outlier_threshold)
                plt.scatter(filtered_pcs[:n, 0], filtered_pcs[:n, 1], s=16, color=self.parent.color_gen.get_color(stype), alpha=0.70)

                # Create a legend entry with a Line2D object to represent the contour's color and label
                legend_handle = mlines.Line2D([], [], color=self.parent.color_gen.get_color(stype), label=stype)
                legend_handles.append(legend_handle)

            # Add title and labels
            title = f'PCA {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.xlabel(f'Principal Component 1 ({self.pcs_var[0]*100:.1f} %)', fontsize=12)
            plt.ylabel(f'Principal Component 2 ({self.pcs_var[1]*100:.1f} %)', fontsize=12)
            plt.legend(handles=legend_handles, fontsize=10, loc='lower right')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_pca_kde(self, outlier_threshold=3.0, save_fig=True):
            def remove_outliers(data, threshold=3.0):
                data = data.astype(float)
                mean = np.mean(data, axis=0)
                std_dev = np.std(data, axis=0)
                mask = np.all(np.abs((data - mean) / std_dev) <= threshold, axis=1)
                return data[mask]
            
            fig = plt.figure(figsize=(6, 6))
            legend_handles = []

            for stype in self.pcs.keys():
                # Filter out outliers
                filtered_pcs = remove_outliers(self.pcs[stype], outlier_threshold)

                # Create a grid for KDE evaluation
                x, y = np.meshgrid(np.linspace(np.min(filtered_pcs[:, 0]), np.max(filtered_pcs[:, 0]), 100),
                                np.linspace(np.min(filtered_pcs[:, 1]), np.max(filtered_pcs[:, 1]), 100))
                positions = np.vstack([x.ravel(), y.ravel()])
                density = np.reshape(self.pcs_kde[stype](positions).T, x.shape)

                # Plot the density contours with 4 levels
                contour = plt.contour(x, y, density, levels=4, linewidths=1, colors=self.parent.color_gen.get_color(stype))

                # Create a legend entry with a Line2D object to represent the contour's color and label
                legend_handle = mlines.Line2D([], [], color=self.parent.color_gen.get_color(stype), label=stype)
                legend_handles.append(legend_handle)

            title = f'PCA KDE {self.parent.title_addition}'
            plt.title(title, fontsize=14)
            plt.xlabel(f'Principal Component 1 ({self.pcs_var[0]*100:.1f} %)', fontsize=12)
            plt.ylabel(f'Principal Component 2 ({self.pcs_var[1]*100:.1f} %)', fontsize=12)
            plt.legend(handles=legend_handles, fontsize=10, loc='lower right')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()


        def plot_correlation_matrix(self, save_fig=True):
            for stype in self.correlation_matrix.keys():
                #cmap = LinearSegmentedColormap.from_list(f"{stype} cmap", ['#C0C0C0', self.parent.color_gen.get_color(stype)])
                cmap = LinearSegmentedColormap.from_list(f"{stype} cmap", ['#C0C0C0', '#000000'])

                fig = plt.figure(figsize=(6, 6))
                plt.imshow(self.correlation_matrix[stype], cmap=cmap, vmin=-1, vmax=1)
                plt.colorbar()
                plt.xlabel('Feature index')
                plt.ylabel('Feature index')
                title = f'Correlation matrix {stype} {self.parent.title_addition}'
                plt.title(title)
                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Dist")
                plt.show()


        def plot_diff_correlation_matrix(self, save_fig=True):
            maxs = {}
            mins = {}
            for stype in self.diff_correlation_matrix.keys():
                maxs[stype] = max(self.diff_correlation_matrix[stype].flatten())
                mins[stype] = min(self.diff_correlation_matrix[stype].flatten())
            maxlim = np.mean(list(maxs.values()))
            minlim = np.mean(list(mins.values()))

            for stype in self.diff_correlation_matrix.keys():
                #cmap = LinearSegmentedColormap.from_list(f"{stype} cmap", [self.parent.color_gen.get_color(stype[1]), self.parent.color_gen.get_color(stype[0])])
                cmap = LinearSegmentedColormap.from_list(f"{stype} cmap", ['#C0C0C0', '#000000'])

                fig = plt.figure(figsize=(6, 6))
                plt.imshow(self.diff_correlation_matrix[stype], cmap=cmap, vmin=minlim, vmax=maxlim)
                plt.colorbar()
                plt.xlabel('Feature index')
                plt.ylabel('Feature index')
                title = f'Difference in correlation matrices {stype[0]}-{stype[1]} {self.parent.title_addition} \n range = [{mins[stype]:.2f} {maxs[stype]:.2f}]'
                plt.title(title)
                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, f'Difference in correlation matrices {stype} {self.parent.title_addition}', dir="Dist") # \n could make problems in save_fig, therefore the title is different from the filename 
                plt.show()
        

        def plot_vector_length(self, hist_stds_to_include=3, focus_stype=0, save_fig=True):
            if focus_stype == 0: focus_stype=list(self.l2_norm.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.l2_norm.keys())[focus_stype]

            hist_range_max = np.mean(self.l2_norm[focus_stype]) + hist_stds_to_include * np.std(self.l2_norm[focus_stype].astype(float))
            hist_range_min = np.mean(self.l2_norm[focus_stype]) - hist_stds_to_include * np.std(self.l2_norm[focus_stype].astype(float))

            fig = plt.figure(figsize=(8, 4))
            for stype in self.l2_norm.keys():
                plt.hist(self.l2_norm[stype][np.logical_and(self.l2_norm[stype] > hist_range_min, self.l2_norm[stype] < hist_range_max)], 
                         bins=100, color=self.parent.color_gen.get_color(stype), alpha=0.2, density=True)
                
                x_values = np.linspace(hist_range_min, hist_range_max, 1000)
                plt.plot(x_values, self.l2_norm_kde[stype](x_values), color=self.parent.color_gen.get_color(stype), label=stype, linestyle='-')

            title = f'Vector length distribution {self.parent.title_addition}'
            plt.title(title)
            plt.ylabel('Density')
            plt.xlabel('L2 Norm')
            plt.xlim([hist_range_min, hist_range_max])
            plt.legend(loc='upper left')
            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Dist")
            plt.show()

        def plot_ttest(self, save_fig=True):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for stype in self.pvalues.keys():
                plt.plot(x, self.pvalues[stype], color=self.parent.color_gen.get_color(stype), label=f'p-value {stype[0]}-{stype[1]}')

            plt.hlines(0.05, x[0], x[-1], color='grey', linestyle='--', label='p = 0.05')

            # Labels and title
            plt.ylabel('p-value')
            plt.yscale('log')
            plt.xlabel('time')
            title = 'Two-tailed p-value of Student t-test'
            plt.title(title)

            # Show legend and plot
            plt.legend(loc='lower right')
            plt.tight_layout()
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
            if self._add_ttest:
                self.plot_ttest(save_fig=save_figs)


    # ----------------- #
    #  SUBCLASS LOSSES  #
    # ----------------- #

    class Loss():
        """
        Loss subclass. Has no functionality, other than plotting loss vs epoch
        """
        def __init__(self, parent, epoch=None):
            self.epoch = epoch if epoch is not None else 'None'
            self.parent = parent
            self._add_losses = False


        def add_losses(self):
            # To keep a uniform structure
            self._add_losses = True


        def plot_losses(self, save_fig=True):
            # check if loss values exist in the loss metric manager
            if len(list(self.parent.loss_metric_manager.losses.values())) != 0:
                fig = plt.figure(figsize=(20, 5))

                # Generate distinct colors for each metric using default color cycle
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                most_loss_entries = max(len(values) for values in self.parent.loss_metric_manager.losses.values())

                ax1 = plt.gca()  # Get the current axis
                ax1.set_xlabel("steps")
                ax1.yaxis.set_ticks([])

                handles = []
                labels = []

                for i, (loss_name, loss_values) in enumerate(self.parent.loss_metric_manager.losses.items()):
                    color = colors[i % len(colors)] 

                    if len(loss_values) > 1:
                        steps = [index * (most_loss_entries - 1) // (len(loss_values) - 1) for index in range(len(loss_values))]
                    else:
                        steps = [0]  # If there's only one element, it corresponds to the first step

                    ax2 = ax1.twinx()
                    ax2.plot(steps, loss_values, color=color, label=loss_name)
                    ax2.spines['right'].set_position(('outward', 60 * (i * 0.8)))  # Offset for visibility
                    ax2.tick_params(axis='y', labelcolor=color)  # Color the y-axis ticks on the right side

                    # Add handles and labels for the legend
                    handles.append(plt.Line2D([0], [0], color=color, lw=2))
                    labels.append(loss_name)

                ax2.legend(handles=handles, labels=labels, loc='upper right')

                title = f"Losses up until epoch {self.epoch}"
                ax1.set_title(title)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Loss")
                plt.show()


        def plot(self, save_figs=True):
            if self._add_losses:
                self.plot_losses(save_fig=save_figs)

    # ----------------- #
    #  SUBCLASS METRIC  #
    # ----------------- #

    class Metric():
        """
        Subclass for determining metrics on the datasets. 
        These inlcude add_:
            metrics: for including metrics outside the class using the loss_metric_manager
            hotelling_score: log of the hotelling statistic
            hotelling_p: p value of the hotelling t^2 test
            roc: using a LogReg classifiert to distinguish between the stypes
            authenticity: authenticity metric for each stype
        and plot_:
            metrics: plots metrics vs epoch
            roc: plots the rocs        
        """
        def __init__(self, parent, epoch=None):
            self.epoch = epoch if epoch is not None else 'None'
            self.parent = parent

            # Booleans for checking whether to automatically create a plot when calling .plot() 
            self._add_metrics = False
            self._add_hotelling_score = False
            self._add_hotelling_p = False
            self._add_roc = False
            self._add_authenticity = False


        def add_metrics(self,):
            self._add_metrics = True


        def add_hotelling_score(self, spectra_cap=600, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']
            
            if not self._add_hotelling_p:
                for stype in stype_combinations:

                    hotelling_results = hotelling_t2(self.parent.spectra[stype[0]][:spectra_cap], self.parent.spectra[stype[1]][:spectra_cap])
                    hotelling_score = np.nan_to_num(np.log(hotelling_results[0]))
                    p_value = np.nan_to_num(np.log(hotelling_results[2]), nan=-256)
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_t2 {stype}': hotelling_score})
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_p {stype}': p_value})

            self._add_hotelling_score = True


        def add_hotelling_p(self, spectra_cap=600, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']
            
            if not self._add_hotelling_score:
                for stype in stype_combinations:

                    hotelling_results = hotelling_t2(self.parent.spectra[stype[0]][:spectra_cap], self.parent.spectra[stype[1]][:spectra_cap])
                    hotelling_score = np.nan_to_num(np.log(hotelling_results[0]))
                    p_value = np.nan_to_num(np.log(hotelling_results[2]), nan=-256)
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_t2 {stype}': hotelling_score})
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_p {stype}': p_value})

            self._add_hotelling_p = True


        def add_roc(self, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            self.roc_results = {}
            
            for stype in stype_combinations:
                X0 = self.parent.spectra_scaled[stype[0]]
                X1 = self.parent.spectra_scaled[stype[1]]

                min_data_length = min(len(X0), len(X1))
                X0 = X0[:min_data_length]
                X1 = X1[:min_data_length]

                y0 = np.array([0]*min_data_length)
                y1 = np.array([1]*min_data_length)

                X = np.concatenate([X0, X1])
                y = np.concatenate([y0, y1])

                self.roc_results[stype] = Logistic_Regression_Classifier(X, y)

            self._add_roc = True


        # ROC curve calculations when using single visit cohorts
        def add_roc_svc(self, split_index_map, tt_split=0.8, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            self.split_masks = {}
            for stype in split_index_map.keys():
                self.split_masks[stype] = {f"mask_{i}": np.array(split_index_map[stype] == i+1)  for i in range(len(np.unique(split_index_map[stype])))}


            self.roc_results = {}

            for stype in stype_combinations:
                    
                tprs = []
                aucs = []
                avg_fpr = np.linspace(0, 1, 100)

                for mask0, mask1, in zip(self.split_masks[stype[0]].values(), self.split_masks[stype[1]].values()):

                    X0 = self.parent.spectra_scaled[stype[0]][mask0]
                    X1 = self.parent.spectra_scaled[stype[1]][mask1]

                    X0 = np.array(X0)
                    X1 = np.array(X1)

                    balanced_data_size = min(len(X0), len(X1))
                    train_data_limit = int(balanced_data_size*tt_split)

                    X0_train = X0[:train_data_limit]
                    X0_test = X0[train_data_limit:]

                    X1_train = X1[:train_data_limit]
                    X1_test = X1[train_data_limit:]

                    X_train = np.vstack([X0_train, X1_train])
                    X_test = np.vstack([X0_test, X1_test])

                    y_train = np.array([0]*len(X0_train)+[1]*len(X1_train))
                    y_test = np.array([0]*len(X0_test)+[1]*len(X1_test))

                    perm_train = np.random.permutation(len(y_train))
                    X_train = X_train[perm_train]
                    y_train = y_train[perm_train]

                    perm_test = np.random.permutation(len(y_test))
                    X_test = X_test[perm_test]
                    y_test = y_test[perm_test]

                    clf = LogisticRegression(penalty='l2', C=10, max_iter=10000)
                    probas = clf.fit(X_train, y_train).decision_function(X_test)
                    fpr, tpr, thresholds = roc_curve(y_test, probas)
                    tprs.append(np.interp(avg_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)

                avg_tpr = np.mean(tprs, axis=0)
                avg_tpr[-1] = 1.0
                avg_auc = auc(avg_fpr, avg_tpr)
                std_auc = np.std(aucs)

                std_tpr = np.std(tprs, axis=0)

                self.roc_results[tuple(stype)] = [avg_fpr, avg_tpr, std_tpr, avg_auc, std_auc]

                self.parent.loss_metric_manager.add_metrics({f'auc {stype}': avg_auc})

            self._add_roc = True


        # manual implementation of authenticity
        def add_authenticity(self, first_half_vs_second_half=False, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            for stype in stype_combinations:
                X0 = self.parent.spectra_scaled[stype[0]]
                X1 = self.parent.spectra_scaled[stype[1]]
                
                if first_half_vs_second_half==True:
                    X0 = X0[:int(len(X0)/2)]
                    X1 = X1[int(len(X1)/2):int(len(X1)/2)*2] # :int(len(X1)/2)*2 is needed for if the length of the dataset is uneven

                authenticity=compute_authenticity(X0, X1)
                self.parent.loss_metric_manager.add_metrics({f'authenticity {stype}': authenticity})

            self._add_authenticity = True

        
        # authenticity for single visit cohorts
        def add_authenticity_svc(self, split_index_map, first_half_vs_second_half=False, **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stype_combinations = handle['stype_combinations']

            self.split_masks = {}
            for stype in split_index_map.keys():
                self.split_masks[stype] = {f"mask_{i}": np.array(split_index_map[stype] == i+1)  for i in range(len(np.unique(split_index_map[stype])))}

            for stype in stype_combinations:
                authenticities = []
                for mask0, mask1, in zip(self.split_masks[stype[0]].values(), self.split_masks[stype[1]].values()):
                    X0 = self.parent.spectra_scaled[stype[0]][mask0]
                    X1 = self.parent.spectra_scaled[stype[1]][mask1]

                    
                    if first_half_vs_second_half==True:
                        X0 = X0[:int(len(X0)/2)]
                        X1 = X1[int(len(X1)/2):int(len(X1)/2)*2] # :int(len(X1)/2)*2 is needed for if the length of the dataset is uneven

                    authenticities.append(compute_authenticity(X0, X1))
                avg_autenticity = np.mean(authenticities)
                self.parent.loss_metric_manager.add_metrics({f'authenticity {stype}': avg_autenticity})

            self._add_authenticity = True


        def plot_metrics(self, save_fig=True):
            # check if metric values exist in the loss metric manager
            if len(list(self.parent.loss_metric_manager.metrics.values())) != 0:
                fig = plt.figure(figsize=(20, 5))

                # Generate distinct colors for each metric using default color cycle
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                most_metric_entries = max(len(values) for values in self.parent.loss_metric_manager.metrics.values())

                ax1 = plt.gca()  # Get the current axis
                ax1.set_xlabel("steps")
                ax1.yaxis.set_ticks([])

                handles = []
                labels = []

                for i, (metric_name, metric_values) in enumerate(self.parent.loss_metric_manager.metrics.items()):
                    color = colors[i % len(colors)] 

                    if len(metric_values) > 1:
                        steps = [index * (most_metric_entries - 1) // (len(metric_values) - 1) for index in range(len(metric_values))]
                    else:
                        steps = [0]  # If there's only one element, it corresponds to the first step

                    ax2 = ax1.twinx()
                    ax2.plot(steps, metric_values, color=color, label=metric_name)
                    ax2.spines['right'].set_position(('outward', 60 * (i * 0.8)))  # Offset for visibility
                    ax2.tick_params(axis='y', labelcolor=color)  # Color the y-axis ticks on the right side

                    # Add handles and labels for the legend
                    handles.append(plt.Line2D([0], [0], color=color, lw=2))
                    labels.append(metric_name)

                ax2.legend(handles=handles, labels=labels, loc='upper right')

                title = f"Metrics up until epoch {self.epoch}"
                ax1.set_title(title)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Metric")
                plt.show()

        
        def plot_roc(self, save_fig=True):
            fig = plt.figure(figsize=(5, 5))

            for stype in self.roc_results.keys():
                avg_fpr, avg_tpr, std_tpr, avg_auc, std_auc = self.roc_results[stype]
                plt.plot(avg_fpr, avg_tpr,label=f"{stype[0]}-{stype[1]} (AUC = {avg_auc:.2f} ± {std_auc:.2f})",color=self.parent.color_gen.get_color(stype), linewidth=2)
                plt.fill_between(avg_fpr, avg_tpr+std_tpr, avg_tpr-std_tpr, label=f'std ROC {stype[0]}-{stype[1]}', color=self.parent.color_gen.get_color(stype), alpha=0.2)

            plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            title = f"ROC between data {self.parent.title_addition}"
            plt.title(title, fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend(loc='lower right', fontsize=10)

            plt.tight_layout()
            if save_fig:
                self.parent.plt_save_fig(fig, title, dir="Metric")
            plt.show()


        def plot(self, save_figs=True):
            if self._add_metrics:
                self.plot_metrics(save_fig=save_figs)
            if self._add_roc:
                self.plot_roc(save_fig=save_figs)



    # -------------------- #
    #  SUBCLASS CONDITION  #
    # -------------------- #

    def add_conditions(self, label_names, label_map):

        self.conditions = defaultdict(lambda: defaultdict(dict))
        self.label_names = label_names
        for stype, labels in label_map.items():
            for col_index, label in enumerate(label_names):
                self.conditions[stype][label] = [row[col_index] for row in labels]

    class Condition():
        """
        Subclass for analysis on the conditions. These include add_ and plot_ for:
            roc
            differential_fingerprint
            effect_size
            threshold_effect
        """

        def __init__(self, parent):
            self.parent = parent

            # Booleans for checking whether to automatically create a plot when calling .plot() 
            self._add_roc = False
            self._add_differential_fingerprint = False
            self._add_effect_size = False
            self._add_correlation = False
            self._add_threshold_effect = False
            self._add_ttest = False

        # ----------------------- #
        #  Calculation functions  #
        # ----------------------- #

        def add_roc(self, labels_to_include='all', tt_split=0.8, **kwargs):
            if labels_to_include == 'all': labels_to_include=self.parent.label_names

            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']
            stype_products = handle['stype_products']

            self.roc_results = defaultdict(lambda: defaultdict(dict))

            self.roc_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))
            self.roc_stypes = stype_products

            for stype in stype_products:
                for label in self.roc_labels:
                    X_stype0 = self.parent.spectra_scaled[stype[0]]
                    X_stype1 = self.parent.spectra_scaled[stype[1]]

                    y_stype0 = np.array(self.parent.conditions[stype[0]][label])
                    y_stype1 = np.array(self.parent.conditions[stype[1]][label])

                    X0_stype0 = X_stype0[y_stype0 == 0]
                    X1_stype0 = X_stype0[y_stype0 == 1]
                    X0_stype1 = X_stype1[y_stype1 == 0]
                    X1_stype1 = X_stype1[y_stype1 == 1]

                    balanced_data_size = min(len(X0_stype0), len(X1_stype0), len(X0_stype1), len(X1_stype1))
                    train_data_limit = int(balanced_data_size*tt_split)

                    X0_train = X0_stype0[:train_data_limit]
                    X1_train = X1_stype0[:train_data_limit]
                    X0_test = X0_stype1[train_data_limit:]
                    X1_test = X1_stype1[train_data_limit:]

                    X_train = np.vstack([X0_train, X1_train])
                    y_train = np.array([0]*len(X0_train) + [1]*len(X1_train))

                    X_test = np.vstack([X0_test, X1_test])
                    y_test = np.array([0]*len(X0_test) + [1]*len(X1_test))

                    perm_train = np.random.permutation(len(y_train))
                    X_train = X_train[perm_train]
                    y_train = y_train[perm_train]

                    perm_test = np.random.permutation(len(y_test))
                    X_test = X_test[perm_test]
                    y_test = y_test[perm_test]

                    self.roc_results[stype][label] = Logistic_Regression_Classifier_def_train_test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        
            self._add_roc = True


        def add_roc_svc(self, split_index_map=None, labels_to_include='all', tt_split=0.8, **kwargs):
            if labels_to_include == 'all': labels_to_include=self.parent.label_names

            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']
            stype_products = handle['stype_products']

            if split_index_map is not None:
                self.split_masks = {}
                for stype in split_index_map.keys():
                    self.split_masks[stype] = {f"mask_{i}": np.array(split_index_map[stype] == i+1)  for i in range(len(np.unique(split_index_map[stype])))}
            else: self.split_masks = self.parent.default_split_masks

            self.roc_results = defaultdict(lambda: defaultdict(dict))
            self.tprs = defaultdict(lambda: defaultdict(dict))

            self.roc_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))
            self.roc_stypes = stype_products
            self.all_aucs = {}

            for stype in stype_products:
                for label in self.roc_labels:

                    tprs = []
                    aucs = []
                    avg_fpr = np.linspace(0, 1, 100)

                    i = 0
                    for mask0, mask1 in zip(self.split_masks[stype[0]].values(), self.split_masks[stype[1]].values()):
                        i += 1
                        X_stype0 = self.parent.spectra_scaled[stype[0]][mask0]
                        X_stype1 = self.parent.spectra_scaled[stype[1]][mask1]

                        y_stype0 = np.array(self.parent.conditions[stype[0]][label])[mask0]
                        y_stype1 = np.array(self.parent.conditions[stype[1]][label])[mask1]

                        X0_stype0 = X_stype0[y_stype0 == 0]
                        X1_stype0 = X_stype0[y_stype0 == 1]
                        X0_stype1 = X_stype1[y_stype1 == 0]
                        X1_stype1 = X_stype1[y_stype1 == 1]

                        balanced_data_size = min(len(X0_stype0), len(X1_stype0), len(X0_stype1), len(X1_stype1))
                        train_data_limit = int(balanced_data_size*tt_split)

                        X0_train = X0_stype0[:train_data_limit]
                        X1_train = X1_stype0[:train_data_limit]
                        X0_test = X0_stype1[train_data_limit:]
                        X1_test = X1_stype1[train_data_limit:]

                        X_train = np.vstack([X0_train, X1_train])
                        y_train = np.array([0]*len(X0_train) + [1]*len(X1_train))

                        X_test = np.vstack([X0_test, X1_test])
                        y_test = np.array([0]*len(X0_test) + [1]*len(X1_test))

                        perm_train = np.random.permutation(len(y_train))
                        X_train = X_train[perm_train]
                        y_train = y_train[perm_train]

                        perm_test = np.random.permutation(len(y_test))
                        X_test = X_test[perm_test]
                        y_test = y_test[perm_test]

                        clf = LogisticRegression(penalty='l2', C=10, max_iter=10000)
                        probas = clf.fit(X_train, y_train).decision_function(X_test)
                        fpr, tpr, thresholds = roc_curve(y_test, probas)
                        tprs.append(np.interp(avg_fpr, fpr, tpr))
                        tprs[-1][0] = 0.0
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)

                        self.all_aucs[f'{stype}-{label}-SVC{i}'] = roc_auc

                    avg_tpr = np.mean(tprs, axis=0)
                    avg_tpr[-1] = 1.0
                    avg_auc = auc(avg_fpr, avg_tpr)
                    std_auc = np.std(aucs)
                    std_tpr = np.std(tprs, axis=0)

                    self.tprs[tuple(stype)][label] = tprs
                    self.roc_results[tuple(stype)][label] = [avg_fpr, avg_tpr, std_tpr, avg_auc, std_auc]

            self._add_roc = True
        

        def add_differential_fingerprint(self, labels_to_include='all', **kwargs):
            if labels_to_include == 'all': labels_to_include=self.parent.label_names

            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            self.diff_fp_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))

            self.diff_fp = defaultdict(lambda: defaultdict(dict))
            self.std_diff_fp = defaultdict(lambda: defaultdict(dict))

            for stype in stypes:
                for label in self.diff_fp_labels:
                    X0 = np.array(self.parent.spectra[stype])
                    X0 = X0[np.array(self.parent.conditions[stype][label])==0]
                    X1 = np.array(self.parent.spectra[stype])
                    X1 = X1[np.array(self.parent.conditions[stype][label])==1]

                    self.diff_fp[stype][label] = np.array(np.mean(X0, axis=0) - np.mean(X1, axis=0), dtype=float)
                    self.std_diff_fp[stype][label] = np.array(np.std(X0.astype(float), axis=0), dtype=float)

            self._add_differential_fingerprint = True


        def add_effect_size(self, labels_to_include='all', convention = 'cohens d', **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            self.effect_size_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))

            self.effect_size = defaultdict(lambda: defaultdict(dict))

            if convention == 'cohens d':
                for stype in stypes:
                    for label in self.effect_size_labels:
                        X0 = np.array(self.parent.spectra[stype])
                        X0 = X0[np.array(self.parent.conditions[stype][label])==0]
                        X1 = np.array(self.parent.spectra[stype])
                        X1 = X1[np.array(self.parent.conditions[stype][label])==1]

                        self.effect_size[stype][label] = cohens_d(X0.astype(float), X1.astype(float))

            elif convention == 'standardized mean difference':
                if self._add_differential_fingerprint == False:
                    self.add_differential_fingerprint()
                for stype in stypes:
                    for label in self.effect_size_labels:
                        self.effect_size[stype][label] = self.diff_fp[stype][label] / self.std_diff_fp[stype][label]

            else:
                raise ValueError('convention must be cohens d or standardized mean difference ')

            self._add_effect_size = True

        def add_correlation(self, labels_to_include='all', convention='spearman', **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']
        
            if isinstance(labels_to_include, str):
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names
        
            self.correlation_labels = matching_list_items(
                labels_to_include,
                filter_for_binary_condition(self.parent.conditions[stypes[0]])
            )
        
            self.correlation = defaultdict(lambda: defaultdict(dict))
        
            for stype in stypes:
                X = np.asarray(self.parent.spectra[stype])  # shape: (n_samples, n_features)
        
                for label in self.correlation_labels:
                    y = np.asarray(self.parent.conditions[stype][label])  # shape: (n_samples,)
        
                    # --- column-wise correlation ---
                    if convention == 'pearson':
                        # vectorized Pearson (fastest)
                        X_centered = X - X.mean(axis=0)
                        y_centered = y - y.mean()
                        numerator = np.sum(X_centered * y_centered[:, None], axis=0)
                        denominator = np.sqrt(
                            np.sum(X_centered**2, axis=0) * np.sum(y_centered**2)
                        )
                        corr = numerator / denominator
        
                    elif convention == 'spearman':
                        # rank transform then Pearson
                        X_ranked = np.apply_along_axis(scipy.stats.rankdata, 0, X)
                        y_ranked = scipy.stats.rankdata(y)
        
                        X_centered = X_ranked - X_ranked.mean(axis=0)
                        y_centered = y_ranked - y_ranked.mean()
                        numerator = np.sum(X_centered * y_centered[:, None], axis=0)
                        denominator = np.sqrt(
                            np.sum(X_centered**2, axis=0) * np.sum(y_centered**2)
                        )
                        corr = numerator / denominator
        
                    elif convention == 'kendall':
                        # no clean vectorization → loop
                        corr = np.array([
                            scipy.stats.kendalltau(X[:, i], y)[0]
                            for i in range(X.shape[1])
                        ])
                    else:
                        raise ValueError(f"Unknown convention: {convention}")
        
                    self.correlation[stype][label] = corr  # shape: (n_features,)
        
            self._add_correlation = True


        def add_ttest(self, labels_to_include='all', **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            self.ttest_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))

            self.pvalues = defaultdict(lambda: defaultdict(dict))

            for stype in stypes:
                for label in self.ttest_labels:
                        X0 = np.array(self.parent.spectra[stype])
                        X0 = X0[np.array(self.parent.conditions[stype][label])==0]
                        X1 = np.array(self.parent.spectra[stype])
                        X1 = X1[np.array(self.parent.conditions[stype][label])==1]

                        _, self.pvalues[stype][label] = ttest_ind(X0, X1, axis=0)
            
            self._add_ttest = True


        def add_threshold_effect(self, min_n_samples=100, labels_to_include='all', **kwargs):
            handle = self.parent.handle_stpye_kwargs(kwargs)
            stypes = handle['stypes']
            stype_combinations = handle['stype_combinations']

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names
            
            def _split_by_condition_threshold(X, y, threshold):
                y = np.array(y)
                if len(y) != X.shape[0]:
                    print('shape of y')
                    print(np.shape(y))
                    print('shape of spectra')
                    print(np.shape(X))
                    raise ValueError("The length of 'y' must match the number of rows in 'X'.")

                below_mask = y < threshold
                above_mask = ~below_mask

                below_threshold = X[below_mask]
                above_threshold = X[above_mask]

                return below_threshold, above_threshold

            def _get_effect_sizes_at_thresholds(X, y, min_n_samples):
                ds = []
                nas = []
                nbs = []

                thresholds = np.unique(y)
                for i in thresholds:
                    _a, _b = _split_by_condition_threshold(X, y, i)
                    _a = _a.astype(float)
                    _b = _b.astype(float)
                    _d = cohens_d(_a, _b)
                    _na, _nb = np.shape(_a)[0], np.shape(_b)[0]
                    nas.append(_na)
                    nbs.append(_nb)
                    ds.append(_d)

                ds = np.array(ds)
                nas = np.array(nas)
                nbs = np.array(nbs)

                split_includes_min_n_samples = np.array(nas>min_n_samples) * np.array(nbs>min_n_samples)
                ds = ds[split_includes_min_n_samples]
                thresholds = np.unique(y)[split_includes_min_n_samples]
                return ds, thresholds

            self.threshold_effect_labels = matching_list_items(labels_to_include, filter_for_continuous_condition(self.parent.conditions[stypes[0]]))

            self.effect_sizes_at_threshold = defaultdict(lambda: defaultdict(dict))
            self.thresholds = defaultdict(lambda: defaultdict(dict))
            self.average_effect_at_threshold = defaultdict(lambda: defaultdict(dict))
            for stype in stypes:
                for label in matching_list_items(labels_to_include, filter_for_continuous_condition(self.parent.conditions[stype])):
                    self.effect_sizes_at_threshold[stype][label], self.thresholds[stype][label] = _get_effect_sizes_at_thresholds(self.parent.spectra[stype], self.parent.conditions[stype][label], min_n_samples)
                    self.average_effect_at_threshold[stype][label] = np.mean(np.abs(self.effect_sizes_at_threshold[stype][label]), axis=1)

            self.mae_effect_at_treshold = defaultdict(lambda: defaultdict(dict))
            for stype in stype_combinations:
                labels0 = matching_list_items(labels_to_include, filter_for_continuous_condition(self.parent.conditions[stype[0]]))
                labels1 = matching_list_items(labels_to_include, filter_for_continuous_condition(self.parent.conditions[stype[1]]))
                labels = list(set(labels0) & set(labels1))
                for label in labels:
                    self.mae_effect_at_treshold[stype][label] = np.mean(np.abs(self.effect_sizes_at_threshold[stype[0]][label] - self.effect_sizes_at_threshold[stype[1]][label]), axis=1)

            self._add_threshold_effect =True


        # ----------------------- #
        #     Plot functions      #
        # ----------------------- #
                
        def plot_roc(self, group_by=None, save_fig=True):
            #group_by = 'trainset', 'testset', 'None'
            if group_by is None:
                stypes_group = np.expand_dims(self.roc_stypes, axis=1)
            elif group_by == 'trainset':
                stypes_group = group_tuples_by_first_element(self.roc_stypes)
            elif group_by == 'testset':
                stypes_group = group_tuples_by_last_element(self.roc_stypes)
            else:
                stypes_group = group_by
            
            for label in self.roc_labels:
                for group in stypes_group:
                    fig = plt.figure(figsize=(5, 5))
                    for stype in group:
                        mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc = self.roc_results[tuple(stype)][label] 
                        plt.plot(mean_fpr, mean_tpr, label=f"{stype[0]}-{stype[1]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})", linewidth=2, color=self.parent.color_gen.get_color(stype))
                        plt.fill_between(mean_fpr, mean_tpr+std_tpr, mean_tpr-std_tpr,color=self.parent.color_gen.get_color(stype), alpha=0.2)

                    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")

                    plt.xlabel("False Positive Rate", fontsize=12)
                    plt.ylabel("True Positive Rate", fontsize=12)
                    title = f"ROC {label} Classification {self.parent.title_addition}"
                    plt.title(title, fontsize=14)
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.legend(loc='lower right', fontsize=10)

                    plt.tight_layout()
                    if save_fig:
                        self.parent.plt_save_fig(fig, title + str(group), dir="Condition")
                    plt.show()
        

        def plot_differential_fingerprint(self, save_fig=True):
            x = self.parent.vec
            for label in self.diff_fp_labels:
                fig = plt.figure(figsize=(8, 4.8))
                for stype in self.diff_fp.keys():
                    plt.plot(x, self.diff_fp[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))
                    plt.fill_between(
                        x,
                        self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                        self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                        color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std {stype}')

                plt.xlabel("time", fontsize=12)
                plt.ylabel("Difference in Mean", fontsize=12)
                title = f"Differential Fingerprints {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()
        
            
        def plot_effect_size(self, save_fig=True):
            x = self.parent.vec

            for label in self.effect_size_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                for stype in self.effect_size.keys():
                    plt.plot(x, self.effect_size[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.effect_size[stype][label]))
                    ymaxs.append(max(self.effect_size[stype][label]))

                plt.xlabel("time", fontsize=12)
                plt.ylabel("Effect size", fontsize=12)
                title = f"Effect size {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                plt.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()

        
        def plot_correlation(self, save_fig=True):
            x = self.parent.vec

            for label in self.correlation_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                for stype in self.correlation.keys():
                    plt.plot(x, self.correlation[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.correlation[stype][label]))
                    ymaxs.append(max(self.correlation[stype][label]))

                plt.xlabel("time", fontsize=12)
                plt.ylabel("Correlation", fontsize=12)
                title = f"Effect size {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                plt.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()


        def plot_ttest(self, save_fig=True):
            x = self.parent.vec

            for label in self.ttest_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                for stype in self.pvalues.keys():
                    plt.plot(x, self.pvalues[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.pvalues[stype][label]))
                    ymaxs.append(max(self.pvalues[stype][label]))

                plt.hlines(0.05, x[0], x[-1], color='grey', linestyle='--', label='p = 0.05')

                plt.xlabel("time", fontsize=12)
                plt.ylabel("p-value", fontsize=12)
                plt.yscale('log')
                title = f"P-values {label} {self.parent.title_addition}"
                plt.title(title, fontsize=14)
                plt.legend(loc='lower right', fontsize=10)

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()
        
        
        def plot_average_threshold_effect(self, save_fig=True):
            x = self.parent.vec

            for label in self.threshold_effect_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                xmins = []
                xmaxs = []
                for stype in self.average_effect_at_threshold.keys():
                    plt.plot(self.thresholds[stype][label], self.average_effect_at_threshold[stype][label], color=self.parent.color_gen.get_color(stype), label=stype)

                    xmins.append(np.min([self.thresholds[stype][label]]))
                    xmaxs.append(np.max([self.thresholds[stype][label]]))

                    ymins.append(np.min([self.average_effect_at_threshold[stype][label]]))
                    ymaxs.append(np.max([self.average_effect_at_threshold[stype][label]]))

                plt.ylabel('Average Effect Size')
                plt.xlabel(f'Treshold {label}')
                title = f'Average effect size per threshold {label} {self.parent.title_addition}'
                plt.title(title)
                plt.grid()

                plt.xlim([min(xmins), max(xmaxs)])
                xticks = [min(xmins)] + list(np.linspace(min(xmins), max(xmaxs), 7)[1:-1]) + [max(xmaxs)]
                plt.xticks(xticks, labels=[f"{int(tick)}" for tick in xticks])

                plt.ylim([min(ymins),max(ymaxs)])
                plt.legend(loc='upper right')

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()
        

        def plot_mae_threshold_effect(self, save_fig=True):
            x = self.parent.vec
            for label in self.threshold_effect_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                xmins = []
                xmaxs = []
                for stype in self.mae_effect_at_treshold.keys():
                    plt.plot(self.thresholds[stype[0]][label], self.mae_effect_at_treshold[stype][label], color=self.parent.color_gen.get_color(stype), label=stype)

                    xmins.append(np.min([self.thresholds[stype[0]][label]]))
                    xmaxs.append(np.max([self.thresholds[stype[0]][label]]))

                    ymins.append(np.min([self.mae_effect_at_treshold[stype][label]]))
                    ymaxs.append(np.max([self.mae_effect_at_treshold[stype][label]]))

                plt.ylabel('MAE')
                plt.xlabel(f'Treshold {label}')
                title = f'Mean Absolute Error of Effect Sizes {label} {self.parent.title_addition}'
                plt.title(title)
                plt.grid()

                plt.xlim([min(xmins), max(xmaxs)])
                xticks = [min(xmins)] + list(np.linspace(min(xmins), max(xmaxs), 7)[1:-1]) + [max(xmaxs)]
                plt.xticks(xticks, labels=[f"{int(tick)}" for tick in xticks])

                plt.ylim([min(ymins),max(ymaxs)])
                plt.legend(loc='upper right')

                plt.tight_layout()
                if save_fig:
                    self.parent.plt_save_fig(fig, title, dir="Condition")
                plt.show()


        def plot(self, save_figs=True):
            if self._add_roc:
                self.plot_roc(save_fig=save_figs)
            if self._add_differential_fingerprint:
                self.plot_differential_fingerprint(save_fig=save_figs)
            if self._add_effect_size:
                self.plot_effect_size(save_fig=save_figs)
            if self._add_correlation:
                self.plot_correlation(save_fig=save_figs)
            if self._add_threshold_effect:
                self.plot_average_threshold_effect(save_fig=save_figs)
                self.plot_mae_threshold_effect(save_fig=save_figs)
            if self._add_ttest:
                self.plot_ttest(save_fig=save_figs)


    def plot(self, save_figs=True):
        self.Dist.plot(save_figs=save_figs)
        self.Loss.plot(save_figs=save_figs)
        self.Metric.plot(save_figs=save_figs)
        self.Condition.plot(save_figs=save_figs)


    # --------------------- #
    # UPDATE LMM AT THE END #
    # --------------------- #

    def Update_Loss_Metric_Manager(self):
        return self.loss_metric_manager
