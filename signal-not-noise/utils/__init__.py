from .plotting import (
    apply_style,
    COLOURS,
    PALETTE,
    plot_distance_distributions,
    plot_explained_variance,
    plot_2d_comparison,
)
from .data_generators import (
    make_patient_data,
    make_low_d_in_high_d,
    make_swiss_roll_data,
    make_two_class_with_noise,
    make_digit_data,
)

__all__ = [
    "apply_style",
    "COLOURS",
    "PALETTE",
    "plot_distance_distributions",
    "plot_explained_variance",
    "plot_2d_comparison",
    "make_patient_data",
    "make_low_d_in_high_d",
    "make_swiss_roll_data",
    "make_two_class_with_noise",
    "make_digit_data",
]
