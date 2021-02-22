import numpy as np
import openml
import pickle
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from proglearn.forest import UncertaintyForest
from sklearn.impute import SimpleImputer

n_estimators = 500
uf_kappa = 1
up_construction_prop = 0.5

n_jobs = 30

clfs = [
    (
        "IRF",
        CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs
            ),
            method="isotonic",
            cv=5,
        ),
    ),
    (
        "SigRF",
        CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs
            ),
            method="sigmoid",
            cv=5,
        ),
    ),
    (
        "UF",
        UncertaintyForest(
            n_estimators=n_estimators,
            tree_construction_proportion=up_construction_prop,
            kappa=uf_kappa,
        ),
    ),
    ("RF", RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)),
]


def train_test(X, y, task_name, nominal_indices):
    cv = 10
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    results_dict = {
        "task": task_name,
        "task_id": task_id,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "y": y,
        "test_indices": [],
        "n_estimators": n_estimators,
        "cv": cv,
        "nominal_features": len(nominal_indices),
        "UF_params": {
            "tree_construction_proportion": up_construction_prop,
            "kappa": uf_kappa,
            "n_estimators": n_estimators,
        },
    }

    numeric_indices = np.asarray(list(set(range(X.shape[1])) - set(nominal_indices)))
    numeric_transformer = SimpleImputer(strategy='median')

    # Do one hot encoding prior to cross validation
    nominal_transformer = Pipeline(steps=[
        # ('onehot', OneHotEncoder(drop='if_binary')),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

    transformers = []
    if len(numeric_indices) > 0:
        transformers += [('numeric', numeric_transformer, numeric_indices)]
    if len(nominal_indices) > 0:
        transformers += [('nominal', nominal_transformer, nominal_indices)]
    preprocessor = ColumnTransformer(transformers=transformers)

    # Store training indices (random state insures consistent across clfs)
    for train_index, test_index in skf.split(X, y):
        results_dict['test_indices'].append(test_index)


    X = OneHotEncoder(drop='if_binary', sparse=False).fit_transform(X)


    for clf_name, clf in clfs:
        pipeline = Pipeline(steps=[
            ('Preprocessor', preprocessor),
            ('Estimator', clf)])

        fold_probas = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pipeline = pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_test)
            fold_probas.append(y_proba)
        
        results_dict[clf_name] = fold_probas

    with open(f"./results_cv10/{task_name}_results_dict.pkl", "wb") as f:
        pickle.dump(results_dict, f)


logging.basicConfig(filename='run_all.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


benchmark_suite = openml.study.get_suite("OpenML-CC18")  # obtain the benchmark suite

for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    task_name = task.get_dataset().name
    # if task_id < 146825:
    #     print(f'Skipping {task_name} ({task_id})')
    #     logging.info(f'Skipping {task_name} ({task_id})')
    #     continue
    # else:
    print(f'Running {task_name} ({task_id})')
    logging.info(f'Running {task_name} ({task_id})')

    X, y = task.get_X_and_y()  # get the data

    nominal_indices = task.get_dataset().get_features_by_type(
        "nominal", [task.target_name]
    )
    try:
        train_test(X, y, task_name, nominal_indices)
    except Exception as e:
        logging.error(f'Test {task_name} ({task_id}) Failed | X.shape={X.shape} | {len(nominal_indices)} nominal indices')
        logging.error(e)

