from math import exp, lgamma, log
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError
from analytics_extensions.nonparametric import _numeric_sample, _average_ranks


def _gamma_q(a, x):
    if x < 0.0 or a <= 0.0:
        raise InvalidAnalysisInputError("Invalid chi-square parameters.")
    if x == 0.0:
        return 1.0
    eps = 1e-14
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * eps:
                break
        p = total * exp(-x + a * log(x) - lgamma(a))
        return float(min(1.0, max(0.0, 1.0 - p)))
    b = x + 1.0 - a
    c = 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1001):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            break
    return float(min(1.0, max(0.0, exp(-x + a * log(x) - lgamma(a)) * h)))


class KruskalWallisTest(AnalysisMethod):
    name = "kruskal_wallis_test"
    category = "statistics"
    version = "1.0"
    description = "Compare two or more independent groups with the Kruskal-Wallis rank test, tie correction, and epsilon-squared effect size."
    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "dataset"
    role_requirements = (("measure", 2),)
    allow_constant_columns = True
    preserve_missing_values = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data")
        if not isinstance(data, dict) or len(data) < 2:
            raise InvalidAnalysisInputError("Kruskal-Wallis requires at least two groups.")
        raw_names = list(data.keys())
        names = [str(n).strip() for n in raw_names]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise InvalidAnalysisInputError("Group names must be unique non-empty strings.")
        groups = []
        stats = {}
        total_count = 0
        missing_count = 0
        for raw, name in zip(raw_names, names):
            values, total, missing = _numeric_sample(data[raw], name)
            groups.append(values)
            stats[name] = {"total_count": total, "sample_size": int(len(values)), "missing_count": missing, "median": float(np.median(values))}
            total_count += total
            missing_count += missing
        combined = np.concatenate(groups)
        ranks, tie_sizes = _average_ranks(combined)
        n = int(len(combined))
        k = len(groups)
        start = 0
        rank_sums = []
        for name, values in zip(names, groups):
            end = start + len(values)
            rs = float(np.sum(ranks[start:end]))
            rank_sums.append(rs)
            stats[name]["rank_sum"] = rs
            stats[name]["mean_rank"] = float(rs / len(values))
            start = end
        raw_h = (12.0 / (n * (n + 1.0))) * sum(rs**2 / len(g) for rs, g in zip(rank_sums, groups)) - 3.0 * (n + 1.0)
        tie_term = float(sum(t**3 - t for t in tie_sizes))
        den = float(n**3 - n)
        tie_correction = 1.0 if den == 0.0 else 1.0 - tie_term / den
        h = 0.0 if tie_correction <= 0.0 else float(max(0.0, raw_h / tie_correction))
        df = k - 1
        p_value = 1.0 if h == 0.0 else _gamma_q(df / 2.0, h / 2.0)
        epsilon_den = n - k        
        epsilon_squared = 0.0 if epsilon_den <= 0 else float(min(1.0, max(0.0, (h - k + 1.0) / epsilon_den)))
        return {"statistic_type": "kruskal_wallis_test", "group_count": k, "total_count": total_count, "sample_size": n, "missing_count": missing_count, "degrees_of_freedom": df, "h_statistic": h, "p_value": p_value, "epsilon_squared": epsilon_squared, "tie_corrected": bool(tie_sizes), "group_statistics": stats, "approximation": "chi_square"}
