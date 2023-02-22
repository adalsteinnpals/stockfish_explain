from stockfish_explain.utils.concepts import (
    evaluate_concepts_one,
    classification_cost,
    value_counts

)
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging
import shelve
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import datetime
import os
import pathlib

file_path = pathlib.Path(__file__).parent.resolve()
experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
experiment_folder = os.path.join(file_path, 'results', experiment_name)
os.mkdir(experiment_folder)


regression_models = [
    Ridge(),
    MLPRegressor(solver='adam', alpha=1e-4, hidden_layer_sizes=(100, 10), random_state=1, max_iter=1000),
    LGBMRegressor()
]

classification_models = [
    RidgeClassifier(),
    MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(100, 10), random_state=1, max_iter=1000, early_stopping=True),
    LGBMClassifier(),
]

concepts = [
    'can_check',
    'can_fork'
 ]

logging.basicConfig(format='%(asctime)s — %(name)s — %(levelname)s — %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler(os.path.join(experiment_folder, "debug.log")),
                        logging.StreamHandler()
                    ],
                    force = True)

log = logging.getLogger('notebook')


metric = classification_cost
db_name = 'stockfish_data_v8_can_fork'

inspect_layer=[10,1,2]
buckets = ['all']
num_splits=5

concept_file = "/media/ap/storage/stockfish_data/concept_table_ballanced_can_fork_white_to_move.csv"


df = pd.read_csv(concept_file)

bucket_values = np.array(df.bucket.values)




variables = {'db_name':db_name, 
            'concept_file':concept_file, 
            'buckets':buckets, 
            'num_splits':num_splits,          
            }
with open(os.path.join(experiment_folder,'parameters.yml'), 'w') as yaml_file:
    yaml.dump(variables, yaml_file, default_flow_style=False)



def concept_probing():

    total_steps = len(concepts) * len(classification_models) * len(inspect_layer) * len(buckets)

    results_classification = []
    with tqdm(total=total_steps) as pbar:
        for target in concepts:
            for model in classification_models:
                for layer in inspect_layer:
                    for bucket in buckets:


                        results = []

                        model_name = f"{model.__class__.__name__}"
                        log.debug(f'starting {model_name} {target} {layer} {bucket}')

                        # Build dataset

                        ## fetch targets
                        y = np.array(df[target].values)
                        if y.dtype == bool:
                            y = y.astype(int)



                        ## Load input
                        sh = shelve.open(f"/media/ap/storage/stockfish_data/{db_name}")
                        X = np.array(sh[f"layer{layer}"])
                        sh.close()
                        X = X.reshape(X.shape[0], -1)

                        log.debug(f'X shape: {X.shape}, y shape {y.shape}')


                        # Remove rows where target is None
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
                            log.debug(f'choosing bucket {bucket}')
                            log.debug(f'X shape: {X.shape}')



                        log.debug(f'Class distributions: {value_counts(y)}')

                        undersample = RandomUnderSampler(
                            sampling_strategy="majority"
                        )
                        # fit and apply the transform
                        X, y = undersample.fit_resample(X, list(y * 1))
                        log.debug('Undersampling...')
                        log.debug(f'X shape: {X.shape}')


                        log.debug(f'Class distributions: {value_counts(y)}')

                        if layer == 10:
                            max_samples = 10000
                            if len(y) > max_samples:
                                # choose 10k random samples from the data
                                _, X, _, y = train_test_split(
                                    X, y, test_size=max_samples, random_state=42
                                )
                            log.debug('Limiting to 10k samples')
                            log.debug(f'X shape: {X.shape}')


                        log.debug(f'Class distributions: {value_counts(y)}')


                        scores  = cross_validate(model, X, y, cv=num_splits, 
                                scoring=make_scorer(metric, greater_is_better=True), return_train_score=True)




                        for score in scores['test_score']:
                            results_ = {
                                "layer": layer,
                                "model_name": model_name,
                                "bucket": bucket,
                                "target": target,
                                "score": score,
                                "training_size": X.shape[0],
                            }

                            results.append(results_)



                        results_classification.append(results)
                        
                        pbar.update(1)

    # pickle results with
    #  timestamp in string
    file_string = os.path.join(experiment_folder, "results.pkl")
    with open(file_string, "wb") as f:
        pkl.dump(results_classification, f)

    print(f'saved file: {file_string}')



if __name__ == '__main__':




    concept_probing()




    # Plot figures

    with open(os.path.join(experiment_folder, 'results.pkl'), 'rb') as file:
        results = pkl.load(file)
        

    # create list from list of lists results
    results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(results)

    layer_name_dict = {1:'Layer 1', 2 : 'Layer 2', 10: 'Input'}
    df['layer_name'] = df.layer.apply(lambda x: layer_name_dict[x])
    # Create bar plot for each layer
    # where x is target, y is score, hue is model_name

    # set matplotlib size
    for target_name in df.target.unique():
        df_ = df[df.target == target_name]
        plt.rcParams['figure.figsize'] = [10, 5]
        sns.barplot(x='layer_name', y='score', hue='model_name', data=df_)
        plt.xticks(rotation=45)
        plt.title(f'Concept: {target_name}')
        plt.grid()
        
        plt.twinx()
        plt.plot(df_.groupby('layer_name').mean().training_size, color='black', marker='o')
        # increase font size
        plt.xticks(fontsize=20)
        plt.ylabel('Training size')
        plt.ylim(0,)
        plt.savefig(os.path.join(experiment_folder, f'concept_{target_name}.pdf'))
        plt.show()