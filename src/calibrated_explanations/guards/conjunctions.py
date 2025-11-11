"""Conjunction validation for perturbation guards."""


def validate_conjunction(x_conj, guard, label_ctx):
    """Validate a combined perturbed point via the guard."""
    return guard.accept(x_conj, label_ctx)
