"""Backward-compatible imports for portfolio tail-risk figures."""

from tailrisk.figures.calibration import (
    SUBPERIOD_EXCEPTION_RATE_PALETTE,
    TAIL_RISK_MODEL_STYLES,
    create_subperiod_exception_interval_figure,
    create_subperiod_exception_rate_figure,
    prepare_subperiod_exception_interval_tables,
    prepare_subperiod_exception_rate_matrices,
    save_subperiod_exception_interval_figure,
    save_subperiod_exception_rate_figure,
)
from tailrisk.figures.inference import (
    create_predictive_inference_figure,
    prepare_predictive_inference_table,
    save_predictive_inference_figure,
)
from tailrisk.figures.monitoring import (
    BASEL_TRAFFIC_LIGHT_PALETTE,
    create_basel_traffic_light_figure,
    save_basel_traffic_light_figure,
)
from tailrisk.figures.predictive import (
    SUBPERIOD_PREDICTIVE_RANK_PALETTE,
    create_subperiod_predictive_first_place_figure,
    create_subperiod_predictive_rank_figure,
    prepare_subperiod_predictive_first_place_counts,
    prepare_subperiod_predictive_rank_matrices,
    save_subperiod_predictive_first_place_figure,
    save_subperiod_predictive_rank_figure,
)
from tailrisk.figures.simulation import (
    save_gaussian_tail_risk_figure,
)


__all__ = [
    "BASEL_TRAFFIC_LIGHT_PALETTE",
    "SUBPERIOD_EXCEPTION_RATE_PALETTE",
    "SUBPERIOD_PREDICTIVE_RANK_PALETTE",
    "TAIL_RISK_MODEL_STYLES",
    "create_basel_traffic_light_figure",
    "create_predictive_inference_figure",
    "create_subperiod_exception_interval_figure",
    "create_subperiod_exception_rate_figure",
    "create_subperiod_predictive_first_place_figure",
    "create_subperiod_predictive_rank_figure",
    "prepare_subperiod_exception_interval_tables",
    "prepare_subperiod_exception_rate_matrices",
    "prepare_predictive_inference_table",
    "prepare_subperiod_predictive_first_place_counts",
    "prepare_subperiod_predictive_rank_matrices",
    "save_basel_traffic_light_figure",
    "save_gaussian_tail_risk_figure",
    "save_predictive_inference_figure",
    "save_subperiod_exception_interval_figure",
    "save_subperiod_exception_rate_figure",
    "save_subperiod_predictive_first_place_figure",
    "save_subperiod_predictive_rank_figure",
]
