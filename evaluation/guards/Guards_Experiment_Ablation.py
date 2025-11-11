# Set environment variable to avoid KMeans memory leak on Windows with MKL
import os
os.environ['OMP_NUM_THREADS'] = '1'

# pylint: disable=invalid-name, line-too-long, duplicate-code
"""
Experiment used to evaluate the stability and robustness of explanations with different guard configurations
"""

import pickle
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from calibrated_explanations import CalibratedExplainer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)


# -------------------------------------------------------
# pylint: disable=invalid-name, missing-function-docstring
def debug_print(message, debug=True):
    if debug:
        print(message)


# ------------------------------------------------------

test_size = 1 / 4  # number of test samples per dataset
is_debug = True
# Guard parameter variations
alphas = [0.05, 0.1, 0.2]  # Significance levels
n_clusters_options = [3, 5, 10]  # Number of clusters per label
covariances = ["diag", "full"]  # Covariance types
use_martingales = [False, True]  # Whether to use martingale test
# Parameters for unguarded comparison (using fast mode)
severities = [0, 0.25, 0.5, 0.75, 1]
noise_type = ["uniform", "gaussian"]


descriptors = [
    "uncal",
    "va",
]  # ,'va'
Descriptors = {"uncal": "Uncal", "va": "VA"}
models = ["xGB", "RF"]  # ['xGB','RF','DT','SVM',] # 'NN',

# pylint: disable=line-too-long
datasets = {
    1: "pc1req",
    2: "haberman",
    3: "hepati",
    4: "transfusion",
    5: "spect",
    6: "heartS",
    7: "heartH",
    8: "heartC",
    9: "je4243",
    10: "vote",
    11: "kc2",
    12: "wbc",
    13: "kc3",
    14: "creditA",
    15: "diabetes",
    16: "iono",
    17: "liver",
    18: "je4042",
    19: "sonar",
    20: "spectf",
    21: "german",
    22: "ttt",
    23: "colic",
    24: "pc4",
    25: "kc1",
}
klara = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
tic_all = time.time()

# -----------------------------------------------------------------------------------------------------
results = {
    "alphas": alphas,
    "n_clusters_options": n_clusters_options,
    "covariances": covariances,
    "use_martingales": use_martingales,
    "severities": severities,  # For unguarded comparison
    "noise_type": noise_type,  # For unguarded comparison
    "test_size": test_size,
}
for dataset in klara:
    dataSet = datasets[dataset]

    tic_data = time.time()
    print(dataSet)
    fileName = "data/" + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=";")
    Xn, y = df.drop("Y", axis=1), df["Y"]

    no_of_classes = len(np.unique(y))
    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    t1 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)  # Changed from min_leaf=4
    t2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)
    s1 = SVC(probability=True)
    s2 = SVC(probability=True)
    r1 = RandomForestClassifier(n_estimators=100)
    r2 = RandomForestClassifier(n_estimators=100)
    h1 = HistGradientBoostingClassifier()
    h2 = HistGradientBoostingClassifier()
    g1 = xgb.XGBClassifier(
        objective="binary:logistic", use_label_encoder=False, eval_metric="logloss"
    )
    g2 = xgb.XGBClassifier(
        objective="binary:logistic", use_label_encoder=False, eval_metric="logloss"
    )

    model_dict = {
        "xGB": (g1, g2, "xGB", Xn),
        "RF": (r1, r2, "RF", Xn),
        "SVM": (s1, s2, "SVM", Xn),
        "DT": (t1, t2, "DT", Xn),
        "HGB": (h1, h2, "HGB", Xn),
    }  # ,'NN': (a1,a2,"NN",Xn)
    model_struct = [model_dict[model] for model in models]
    results[dataSet] = {}
    for c1, c2, alg, X in model_struct:
        tic_algorithm = time.time()
        debug_print(dataSet + " " + alg)
        results[dataSet][alg] = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=test_size, random_state=42
        )
        X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
            X_train, y_train, test_size=1 / 3, random_state=42
        )

        c2.fit(X_prop_train, y_prop_train)
        categorical_features = [
            i for i in range(no_of_features) if len(np.unique(X.iloc[:, i])) < 10
        ]

        ablation = {"ce": [], "pce_guarded": {}, "pce_unguarded": {}, "proba": []}
        abl_timer = {
            "ce_init": [],
            "ce_explain": [],
            "pce_guarded_init": {},
            "pce_guarded_explain": {},
            "pce_unguarded_init": {},
            "pce_unguarded_explain": {},
        }

        # Initialize ablation structures for guard parameters
        for alpha in alphas:
            ablation["pce_guarded"][alpha] = {}
            abl_timer["pce_guarded_init"][alpha] = {}
            abl_timer["pce_guarded_explain"][alpha] = {}
            for n_clusters in n_clusters_options:
                ablation["pce_guarded"][alpha][n_clusters] = {}
                abl_timer["pce_guarded_init"][alpha][n_clusters] = {}
                abl_timer["pce_guarded_explain"][alpha][n_clusters] = {}
                for covariance in covariances:
                    ablation["pce_guarded"][alpha][n_clusters][covariance] = {}
                    abl_timer["pce_guarded_init"][alpha][n_clusters][covariance] = {}
                    abl_timer["pce_guarded_explain"][alpha][n_clusters][covariance] = {}
                    for use_martingale in use_martingales:
                        ablation["pce_guarded"][alpha][n_clusters][covariance][use_martingale] = []
                        abl_timer["pce_guarded_init"][alpha][n_clusters][covariance][use_martingale] = []
                        abl_timer["pce_guarded_explain"][alpha][n_clusters][covariance][use_martingale] = []

        # Initialize for unguarded (using original fast parameters)
        ablation["pce_unguarded"]["none"] = {}
        abl_timer["pce_unguarded_init"]["none"] = {}
        abl_timer["pce_unguarded_explain"]["none"] = {}
        for severity in severities:
            ablation["pce_unguarded"]["none"][severity] = {}
            abl_timer["pce_unguarded_init"]["none"][severity] = {}
            abl_timer["pce_unguarded_explain"]["none"][severity] = {}
            for noise in noise_type:
                ablation["pce_unguarded"]["none"][severity][noise] = []
                abl_timer["pce_unguarded_init"]["none"][severity][noise] = []
                abl_timer["pce_unguarded_explain"]["none"][severity][noise] = []

        tic = time.time()
        ce = CalibratedExplainer(
            c2, X_cal, y_cal, feature_names=df.columns, categorical_features=categorical_features
        )
        ct = time.time() - tic
        abl_timer["ce_init"].append(ct / len(X_cal))

        tic = time.time()
        factual_explanations = ce.explain_factual(X_test)
        ct = time.time() - tic
        abl_timer["ce_explain"].append(ct / len(X_test))
        ablation["ce"].append([f.feature_weights for f in factual_explanations])
        ablation["proba"].append(c2.predict_proba(X_test)[:, 1])

        # Ablation for guarded explanations with different guard parameters
        for alpha in alphas:
            for n_clusters in n_clusters_options:
                for covariance in covariances:
                    for use_martingale in use_martingales:
                        tic = time.time()
                        ce_guarded = CalibratedExplainer(
                            c2,
                            X_cal,
                            y_cal,
                            guard="conformal_regions",
                            guard_params={
                                "alpha": alpha,
                                "n_clusters": n_clusters,
                                "covariance": covariance,
                                "use_martingale": use_martingale,
                                "random_state": 42
                            },
                            feature_names=df.columns,
                            categorical_features=categorical_features,
                            severity=0.5,  # Default severity
                            noise_type="uniform",  # Default noise
                            scale_factor=5,  # Default scale factor
                        )
                        ct = time.time() - tic
                        abl_timer["pce_guarded_init"][alpha][n_clusters][covariance][use_martingale].append(ct / len(X_cal))

                        tic = time.time()
                        explanations = ce_guarded.explain_factual(X_test)
                        ct = time.time() - tic
                        abl_timer["pce_guarded_explain"][alpha][n_clusters][covariance][use_martingale].append(ct / len(X_test))
                        ablation["pce_guarded"][alpha][n_clusters][covariance][use_martingale].append(
                            [f.feature_weights for f in explanations]
                        )

        # Ablation for unguarded explanations (using original fast parameters)
        for severity in severities:
            for noise in noise_type:
                tic = time.time()
                ce_unguarded = CalibratedExplainer(
                    c2,
                    X_cal,
                    y_cal,
                    fast=True,  # Use fast mode for unguarded
                    feature_names=df.columns,
                    categorical_features=categorical_features,
                    severity=severity,
                    noise_type=noise,
                    scale_factor=5,  # Default scale factor
                )
                ct = time.time() - tic
                abl_timer["pce_unguarded_init"]["none"][severity][noise].append(ct / len(X_cal))

                tic = time.time()
                explanations = ce_unguarded.explain_fast(X_test)
                ct = time.time() - tic
                abl_timer["pce_unguarded_explain"]["none"][severity][noise].append(ct / len(X_test))
                ablation["pce_unguarded"]["none"][severity][noise].append(
                    [f.feature_weights for f in explanations]
                )
                # print('')

        results[dataSet][alg]["ablation"] = ablation
        results[dataSet][alg]["timer"] = abl_timer

    toc_data = time.time()
    debug_print(dataSet + ": " + str(toc_data - tic_data), is_debug)
    with open("evaluation/results_guards_ablation.pkl", "wb") as f:
        pickle.dump(results, f)
    # pickle.dump(results, open('evaluation/results_stab_rob.pkl', 'wb'))

toc_all = time.time()
debug_print(str(toc_data - tic_data), is_debug)
