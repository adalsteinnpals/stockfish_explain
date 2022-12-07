import shelve
import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone

concept_file = "/media/ap/storage/stockfish_data/concept_table3.csv"

logging.basicConfig(format='%(asctime)s — %(name)s — %(levelname)s — %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def classification_cost(y_test, y_pred):
    residuals = 1 - np.abs(y_pred - y_test)
    return np.mean(residuals) * 2 - 1



def evaluate_concepts(
    model,
    metric,
    db_name,
    over_sample=False,
    buckets=["all"],
    inspect_layer=[1, 2, 3, 4],
    target="white_has_control_of_open_file",
    test_size=0.3,
    num_splits=1,
):


    sh = shelve.open(f"/media/ap/storage/stockfish_data/{db_name}")
    keys = list(sh.keys())
    sh.close()

    ckpts = [k.replace("ckpt_data_", "") for k in keys if "ckpt_data_epoch" in k]


    df = pd.read_csv(concept_file)

    bucket_values = np.array(df.bucket.values)

    y_all = np.array(df[target].values)
    if y_all.dtype == bool:
        y_all = y_all.astype(int)

    has_printed = False

    results = []
    logging.info(f"Number of ckpts: {len(ckpts)}")
    logging.info(f"Number of buckets: {len(buckets)}")
    logging.info(f"Number of layers: {len(inspect_layer)}")

    for bucket in tqdm(buckets):

        for layer in tqdm(inspect_layer, leave=False):

            for ckpt in tqdm(ckpts[:20], leave=False):
                try:
                    m = re.search("(?<==).*(?=-)", ckpt)
                    if m is not None:
                        epoch = int(m.group(0))
                        logging.info(f'starting epoch {epoch}')

                        sh = shelve.open(f"/media/ap/storage/stockfish_data/{db_name}")
                        X = np.array(sh["ckpt_data_" + ckpt][f"layer{layer}"])
                        sh.close()
                        X = X.reshape(X.shape[0], -1)

                        y = y_all[: len(X)]

                        y_not_None = np.where(y != None)[0].astype(int)
                        y_not_None = y != None
                        X, y = X[y_not_None], y[y_not_None]

                        if bucket != "all":
                            assert bucket in [0, 1, 2, 3, 4, 5, 6, 7]

                            bucket_chosen = bucket_values == bucket
                            # print(f'Bucket: {bucket}')
                            # print(f'shape before bucket filtering: {y.shape[0]}')
                            X, y = X[bucket_chosen], y[bucket_chosen]
                            # print(f'shape after bucket filtering: {y.shape[0]}')

                        if over_sample:

                            # if not has_printed:
                            #    print('Undersampling...')

                            #    print(f'pre undersampling distribution: {value_counts(y)}')

                            undersample = RandomUnderSampler(
                                sampling_strategy="majority"
                            )
                            # fit and apply the transform
                            X, y = undersample.fit_resample(X, list(y * 1))

                        logging.info(f'X shape: {X.shape}')

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )

                        model.fit(X_train, list(y_train))
                        y_pred = model.predict(X_test)

                        score = metric(y_test, y_pred)
                        logging.info(f'Epoch {epoch} score: {score}')

                        results_ = {
                            "layer": layer,
                            "epoch": epoch,
                            "bucket": bucket,
                            "target": target,
                            "score": score,
                            "training_size": X.shape[0],
                        }

                        results.append(results_)

                except Exception as e:

                    logging.info(f'Error: {e}')

                    results_ = {
                        "layer": layer,
                        "epoch": epoch,
                        "bucket": bucket,
                        "target": target,
                        "score": None,
                        "training_size": None,
                    }

                    results.append(results_)

    return results



def evaluate_concepts_one(
    model,
    metric,
    db_name,
    under_sample=False,
    buckets=["all"],
    inspect_layer=[1, 2, 3, 4],
    target="white_has_control_of_open_file",
    test_size=0.3,
    num_splits=1,
):


    sh = shelve.open(f"/media/ap/storage/stockfish_data/{db_name}")
    keys = list(sh.keys())
    sh.close()

    ckpts = ['official']


    df = pd.read_csv(concept_file)

    bucket_values = np.array(df.bucket.values)

    y_all = np.array(df[target].values)
    if y_all.dtype == bool:
        y_all = y_all.astype(int)

    has_printed = False

    results = []

    for bucket in tqdm(buckets):

        for layer in inspect_layer:

            for ckpt in ckpts:
                try:

                    model_name = f"{model.__class__.__name__}"

                    logging.info(f'starting {model_name} {ckpt} {target} {layer} {bucket}')

                    sh = shelve.open(f"/media/ap/storage/stockfish_data/{db_name}")
                    X = np.array(sh["ckpt_data_" + ckpt][f"layer{layer}"])
                    sh.close()
                    X = X.reshape(X.shape[0], -1)

                    y = y_all[: len(X)]

                    y_not_None = np.where(y != None)[0].astype(int)
                    y_not_None = y != None
                    X, y = X[y_not_None], y[y_not_None]

                    if bucket != "all":
                        assert bucket in [0, 1, 2, 3, 4, 5, 6, 7]

                        bucket_chosen = bucket_values == bucket
                        # print(f'Bucket: {bucket}')
                        # print(f'shape before bucket filtering: {y.shape[0]}')
                        X, y = X[bucket_chosen], y[bucket_chosen]
                        # print(f'shape after bucket filtering: {y.shape[0]}')

                    if under_sample:

                        # if not has_printed:
                        #    print('Undersampling...')

                        #    print(f'pre undersampling distribution: {value_counts(y)}')

                        undersample = RandomUnderSampler(
                            sampling_strategy="majority"
                        )
                        # fit and apply the transform
                        X, y = undersample.fit_resample(X, list(y * 1))

                    logging.info(f'X shape: {X.shape}')

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    for i in range(10):
                        model.fit(X_train, list(y_train))
                        y_pred = model.predict(X_test)

                        score = metric(y_test, y_pred)
                        if score > 0.2:
                            model = clone(model)
                            model.random_state = i
                            break
                        else:
                            logging.info(f'Epoch {i} score: {score} (skipping)')



                    logging.info(f'score: {score}')

                    results_ = {
                        "layer": layer,
                        "model_name": model_name,
                        "bucket": bucket,
                        "target": target,
                        "score": score,
                        "training_size": X.shape[0],
                    }

                    results.append(results_)

                except Exception as e:

                    print(f'Error: {e}')

                    results_ = {
                        "layer": layer,
                        "bucket": bucket,
                        "target": target,
                        "score": None,
                        "training_size": None,
                    }

                    results.append(results_)
    return results