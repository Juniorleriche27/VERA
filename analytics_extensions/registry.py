from analytics_extensions.anova import OneWayANOVA
from analytics_extensions.nonparametric import MannWhitneyUTest
from analytics_extensions.kruskal import KruskalWallisTest

HACKATHON_ANALYTICS = (
    OneWayANOVA,
    MannWhitneyUTest,
    KruskalWallisTest,
)


def available_hackathon_methods():
    return [method_class().metadata() for method_class in HACKATHON_ANALYTICS]
