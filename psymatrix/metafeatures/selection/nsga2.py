import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from psymatrix.ranking import eval_net
from psymatrix.utils import load_metafeatures

data = load_metafeatures(
    "emnlp24_datasets.txt", config="base", fillna=0, include_datasets_ids=False
)


def to_feature_list(x, features):
    return [features[i] for i in range(len(x)) if x[i] > 0.5]


class MetafeatureSelectionProblem(ElementwiseProblem):

    def __init__(self):

        self.metafeatures = data.columns.tolist()

        n_var = len(self.metafeatures)

        super().__init__(
            n_var=n_var,
            n_obj=2,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

    def _evaluate(self, x, out, *args, **kwargs):

        features = to_feature_list(x, self.metafeatures)

        # area under the curve
        f1 = eval_net(
            data,
            features=features,
            embedding_size=4,
            embedding_method="pca",
        )

        # number of selected features
        f2 = len(features)

        out["F"] = [-f1, f2]


def run():

    problem = MetafeatureSelectionProblem()

    algorithm = NSGA2(
        pop_size=64,
        n_offsprings=32,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", 200)

    res = minimize(
        problem, algorithm, termination, seed=1, save_history=True, verbose=True
    )

    X = res.X
    F = res.F

    print(X)
    print(F)


with open("metafeatures.nsga2.1.config", "w", encoding="utf8") as f:
    for i, feature in enumerate(to_feature_list(X[-1], data.columns)):
        f.write(f"{feature}")
        if i < len(to_feature_list(X[-1], data.columns)) - 1:
            f.write("\n")

    # TODO: select and save the best features


if __name__ == "__main__":
    run()
