from analytics_extensions.anova import OneWayANOVA
from analytics_extensions.nonparametric import MannWhitneyUTest
from analytics_extensions.kruskal import KruskalWallisTest
from analytics_extensions.multiple_testing import MultipleTestingCorrection
from analytics_extensions.extreme_value import ExtremeValueTailRisk
from analytics_extensions.causal_impact import BayesianStructuralCausalImpact

HACKATHON_ANALYTICS = (
    OneWayANOVA,
    MannWhitneyUTest,
    KruskalWallisTest,
    MultipleTestingCorrection,
    ExtremeValueTailRisk,
    BayesianStructuralCausalImpact,
)


def available_hackathon_methods():
    return [method_class().metadata() for method_class in HACKATHON_ANALYTICS]
