"""
Multi-Output Side-Channel Attack Implementation.

This package implements the reproduction of:
"Efficient Nonprofiled Side-Channel Attack Using Multi-Output Classification Neural Network"
"""

from .data_preprocessing import (
    create_multi_output_labels,
    inject_gaussian_noise,
    apply_desynchronization,
    normalize_traces,
    load_ascad_dataset,
    load_chipwhisperer_dataset,
    create_noisy_dataset,
    create_desync_dataset
)

from .mlp_mo import MLP_MO, MultiOutputLoss, compute_branch_accuracy
from .cnn_mo import CNN_MO
from .training import (
    PowerTraceDataset,
    TrainingLogger,
    train_mlp_mo,
    train_cnn_mo
)
from .evaluation import (
    compute_success_rate,
    compute_key_ranking,
    evaluate_model,
    run_repeated_attacks,
    compare_with_baseline
)
from .visualization import (
    plot_accuracy_curves,
    plot_attack_time_comparison,
    plot_success_rate_comparison,
    plot_branch_accuracies,
    plot_noise_comparison,
    plot_desync_comparison
)

__all__ = [
    # Data preprocessing
    'create_multi_output_labels',
    'inject_gaussian_noise',
    'apply_desynchronization',
    'normalize_traces',
    'load_ascad_dataset',
    'load_chipwhisperer_dataset',
    'create_noisy_dataset',
    'create_desync_dataset',
    # Models
    'MLP_MO',
    'CNN_MO',
    'MultiOutputLoss',
    'compute_branch_accuracy',
    # Training
    'PowerTraceDataset',
    'TrainingLogger',
    'train_mlp_mo',
    'train_cnn_mo',
    # Evaluation
    'compute_success_rate',
    'compute_key_ranking',
    'evaluate_model',
    'run_repeated_attacks',
    'compare_with_baseline',
    # Visualization
    'plot_accuracy_curves',
    'plot_attack_time_comparison',
    'plot_success_rate_comparison',
    'plot_branch_accuracies',
    'plot_noise_comparison',
    'plot_desync_comparison'
]





