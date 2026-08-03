"""Public API for clinical KG retrieval and dataset augmentation."""
from .kg_loader import ClinicalKG
from .context_builder import build_kg_context
from .dataset_builder import build_augmented_dataset
from .coverage_stats import compute_coverage, print_coverage_stats

__all__ = [
    'ClinicalKG',
    'build_kg_context',
    'build_augmented_dataset',
    'compute_coverage',
    'print_coverage_stats'
]
