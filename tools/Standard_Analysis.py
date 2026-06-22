from .calculation_functions import *
from .Evaluator_FTIR import *
from .Evaluator_Peak_Ratios import *


def Standard_Analysis(spectra_map, label_map, label_names ,scaler, split_index_map, vec=None, lmm=None, epoch=None, folder='./'):

    evt = Evaluator_FTIR(spectra_map, scaler=scaler, vec=vec, directory=f'{folder}/Spectra', loss_metric_manager=lmm)
    #evt.figure_settings(title_addition=f'Epoch {epoch}')
    evt.add_conditions(label_names=label_names, label_map=label_map)

    evt.dist.add_sample_spectrum()
    evt.dist.add_sample_spectra()
    evt.dist.add_sample_distribution()
    evt.dist.add_differential_fingerprint()
    evt.dist.add_effect_size()
    evt.dist.add_kde()
    evt.dist.add_pca()
    evt.dist.add_diff_correlation_matrix()
    evt.dist.add_vector_length()
    evt.dist.plot()

    evt.metric.add_roc(split_index_map=split_index_map)
    evt.metric.add_authenticity()
    evt.metric.add_hotelling_p()
    evt.metric.add_hotelling_score()
    evt.metric.add_metrics()
    evt.metric.plot()

    #evt.loss.add_losses()
    #evt.loss.plot()

    evt.condition.add_roc(split_index_map=split_index_map)
    #evt.condition.add_auc_difference()
    evt.condition.add_differential_fingerprint()
    evt.condition.add_effect_size()
    evt.condition.add_threshold_effect()
    evt.condition.plot()

    lmm = evt.Update_Loss_Metric_Manager()


    evt_pr = Evaluator_Peak_Ratios(spectra_map, scaler=scaler, vec=vec, directory=f'{folder}/PeakRatios')
    #evt_pr.figure_settings(title_addition=f'Epoch {epoch}')
    evt_pr.add_conditions(label_names=label_names, label_map=label_map)

    evt_pr.dist.add_sample_spectrum()
    evt_pr.dist.add_sample_spectra()
    evt_pr.dist.add_sample_distribution(wavenumber_index_to_look_at=5)
    evt_pr.dist.add_differential_fingerprint()
    evt_pr.dist.add_effect_size()
    evt_pr.dist.add_kde(inv_frequency = 1)
    evt_pr.dist.add_pca()
    evt_pr.dist.add_diff_correlation_matrix()
    evt_pr.dist.add_vector_length()
    evt_pr.dist.plot()

    evt_pr.metric.add_roc(split_index_map=split_index_map)
    evt_pr.metric.add_authenticity()
    evt_pr.metric.add_hotelling_p()
    evt_pr.metric.add_hotelling_score()
    evt_pr.metric.add_metrics()
    evt_pr.metric.plot()

    #ev_prt.loss.add_losses()
    #ev_prt.loss.plot()

    evt_pr.condition.add_roc(split_index_map=split_index_map)
    #ev_prt.condition.add_auc_difference()
    evt_pr.condition.add_differential_fingerprint()
    evt_pr.condition.add_effect_size()
    evt_pr.condition.add_threshold_effect()
    evt_pr.condition.plot()

    return lmm