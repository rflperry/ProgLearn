import numpy as np
import openml
import pickle
import logging
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from proglearn.forest import UncertaintyForest
from sklearn.impute import SimpleImputer

MODE = 'APPEND'

n_estimators = 500
uf_kappa = 1
up_construction_prop = 0.5
n_jobs = 30
cv = 10

clfs = [
    (
        "IRF",
        CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(
                n_estimators=n_estimators // 5, n_jobs=n_jobs
            ),
            method="isotonic",
            cv=5,
        ),
    ),
    (
        "SigRF",
        CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(
                n_estimators=n_estimators // 5, n_jobs=n_jobs
            ),
            method="sigmoid",
            cv=5,
        ),
    ),
    # (
    #     "UF",
    #     UncertaintyForest(
    #         n_estimators=n_estimators,
    #         tree_construction_proportion=up_construction_prop,
    #         kappa=uf_kappa,
    #     ),
    # ),
    # ("RF", RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)),
]

def _check_nested_equality(lst1, lst2):
    if isinstance(lst1, list) and isinstance(lst2, list):
        for l1, l2 in zip(lst1, lst2):
            if not _check_nested_equality(l1, l2):
                return False
    elif isinstance(lst1, np.ndarray) and isinstance(lst2, np.ndarray):
        return np.all(lst1 == lst2)
    else:
        return lst1 == lst2
    
    return True

def train_test(X, y, task_name, nominal_indices):
    save_path = f"./results_cv{cv}/{task_name}_results_dict.pkl"
    if MODE == 'APPEND':
        if not os.path.isfile(save_path):
            logging.info(f'APPEND MODE: Skipping {task_name}')
            return

    # Set up Cross validation
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Check if existing experiments
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
        "UF_metadata": {
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
        if not f"{clf_name}_metadata" in results_dict.keys():
            results_dict[f"{clf_name}_metadata"] = {}
        results_dict[f"{clf_name}_metadata"]["train_times"] = []
        results_dict[f"{clf_name}_metadata"]["test_time"]= []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start_time = time.time()
            pipeline = pipeline.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_proba = pipeline.predict_proba(X_test)
            test_time = time.time() - (train_time + start_time)

            fold_probas.append(y_proba)
            results_dict[f"{clf_name}_metadata"]["train_times"].append(train_time)
            results_dict[f"{clf_name}_metadata"]["test_time"].append(test_time)
        
        results_dict[clf_name] = fold_probas

    # If existing data, load and append to. Else save
    if os.path.isfile(save_path):
        logging.info(f'Existing data for {task_name} ({task_id}), appending')
        with open(save_path, "rb") as f:
            prior_results = pickle.load(f)

        # Check these keys have the same values
        verify_keys = [
            "task", "task_id", "n_samples", "n_features", "n_classes", "y",
            "test_indices", "n_estimators", "cv", "nominal_features"]
        for key in verify_keys:
            assert _check_nested_equality(prior_results[key], results_dict[key]), key

        # Replace/add data
        replace_keys = [name for name, _ in clfs]
        replace_keys += [f"{name}_metadata" for name in replace_keys]
        for key in replace_keys:
            prior_results[key] = results_dict[key]

        results_dict = prior_results

    with open(save_path, "wb") as f:
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
    print(f'{MODE} {task_name} ({task_id})')
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
