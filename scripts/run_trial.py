"""Run a reproducible trial: 5 random samples with seed 42 (no prompts)."""

from cpt_categorizer.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(sample_n=5, random_seed=42, debug=True)
