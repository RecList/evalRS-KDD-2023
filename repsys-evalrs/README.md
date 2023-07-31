## Implementation of [RepSys](https://github.com/cowjen01/repsys/) for EvalRS 2023 workshop

### I just want to try RepSys without editing the dataset and models:
We have a docker image that contains the implementation of the EvalRS 2023 dataset along with the parquet files and trained models:
```
docker run --rm --name repsys -p 3001:3001 kasape/repsys:kdd-with-data
```

### I want to play with the EvalRS dataset using RepSys and alter the dataset or define a new model and visualize it in RepSys:

**Clone this repository and use its root folder to perform all commands in this guide:**
```
git clone git@github.com:RecList/evalRS-KDD-2023.git
cd evalRS-KDD-2023
```
**Download the dataset EvalRS KDD (1.60GB) and unzip it:**
```
wget https://evalrs.object.lga1.coreweave.com/evalrs_dataset_KDD_2023.zip
unzip evalrs_dataset_KDD_2023.zip 
```
**The implementation of the EvalRS dataset for RepSys expects the data to be found at `repsys-evalrs/data`.**
```
mv evalrs_dataset_KDD_2023 repsys-evalrs/data
```
*Note: For simplicity, we move all the files, but in fact, RepSys uses only `evalrs_emotion_tags.parquet.parquet`, `evalrs_events.parquet`, `evalrs_social_tags.parquet` and `evalrs_tracks.parquet`.*

**RepSys can be run as a docker image with the implementation of the EvalRS 2023 dataset and its data mounted to a folder `/app` by adding `-v ./repsys-evalrs:/app` to docker run command:**

However, before we can use RepSys to vizualize the data, we need to call RepSys CLI to: 
1. Split the dataset into training and validation parts using the following command. A pruning specified in the `repsys-evalrs/repsys.ini` is applied, as described [in the documentation](https://github.com/cowjen01/repsys/tree/master#repsysini). As a result of this step, folder `repsys-evalrs/.repsys_checkpoints` is created (if not existed before) and it contains `dataset-split-[timestamp].zip` file. 
```
docker run --rm -v ./repsys-evalrs:/app kasape/repsys:kdd dataset split
```

2. Train all models in `repsys-evalrs/models/*.py` files or specify a model or specify model appending `-m [model_name]` to the following command:

```
docker run --rm -v ./repsys-evalrs:/app kasape/repsys:kdd model train
```

3. Evalute all models (or specify one using `-m [model_name]`) producing files `repsys-evalrs/.repsys_checkpoints/model-eval-[model-name]-[timestamp].zip`
```
docker run --rm -v ./repsys-evalrs:/app kasape/repsys:kdd model eval
```

4. Evalute the dataset producing a file `repsys-evalrs/.repsys_checkpoints/dataset-eval-[model-name]-[timestamp].zip` with optionally specified method as described in [the documentation](https://github.com/cowjen01/repsys/#evaluating-the-dataset)
```
docker run --rm -v ./repsys-evalrs:/app kasape/repsys:kdd dataset eval
```

**Once we have following files in `repsys-evalrs/.repsys_checkpoints` folder:**
```
repsys-evalrs
├── .repsys_checkpoints
│   ├── dataset-eval-1690727153.zip
│   ├── dataset-split-1690725287.zip
│   ├── elsa.pth
│   ├── model-eval-elsa-1690733672.zip
│   ├── model-eval-knn-1690733672.zip
│   ├── model-eval-pop-1690733672.zip
│   ├── model-eval-rand-1690733672.zip
├── models
│   ├── __init__.py
│   ├── base.py
│   ├── elsa.py
│   ├── random.py
│   ├── pop.py
│   ├── knn.py
├── data
│   ├── evalrs_emotion_tags.parquet
│   ├── evalrs_events.parquet
│   ├── evalrs_social_tags.parquet
│   ├── evalrs_tracks.parquet
├── dataset.py
└── repsys.ini
```
**Repsys can be run as a docker container with:**
```
docker run --rm -v ./repsys-evalrs:/app -p 3001:3001 kasape/repsys:kdd
```

#### Troubleshooting

###### Some of the 4.X steps (model training, model evaluation, dataset evaluation) are taking forever.

For a model or dataset evaluation, you can set higher pruning (`min_item_interacts` and `min_user_interacts`) to keep only items and users that are really important in the dataset to speed up the computation. After editing pruning, the new split needs to be created (Step 4.1). For a model training, you can configure models to be smaller, e.g. alter `repsys-evalrs/models/elsa.py` to have smaller latent space size (`n_dims=64`) or smaller number of epochs.

###### I have edited `repsys-evalrs/dataset.py` to have different items columns or load extra information in `load_items` method, but when I run RecSys, nothing changes.

RecSys needs to preprocess the dataset into its own format and map all the columns of items. Please process to step 4.1.

###### I want to add custom python package to `kasape/repsys:kdd` docker image and use it in dataset loading or in my own model.

You can edit `repsys-evalrs/requirements.txt` and build your version of `kasape/repsys:kdd` using provided `repsys-evalrs/Dockerfile` by calling `docker build -t kasape/repsys:kdd-mine -f Dockerfile .` inside the `repsys-evalrs` folder.

###### I don't have docker, but I want to try RepSys.

RepSys is also possible to use as python a package installed from pypi. Please see the [official RepSys repository](https://github.com/cowjen01/repsys/) for more instructions.

###### I've setup PyMDE as `embed_method` in `repsys.ini` or I've called `dataset evaluate --method pymde` and I'm getting ` ModuleNotFoundError: No module named 'pymde'`.

PyMDE is not included in the original list of requirements.txt for RepSys, because it was making the docker image unnecessarily large. If you want to use it, you can add `pymde==0.1.14` as specified in **I want to add custom python package**, or follow the [instructions](https://github.com/cowjen01/repsys/#installation) if you are installing RepSys from pypi. 

###### I'm getting `Invalid value for '-c' / '--config': Path 'repsys.ini' does not exist`

Folder *repsys-evalrs* wasn't properly mounted to docker container and RepSys framework cannot find `/app/repsys.ini` file.

### I want to modify core of RepSys to make it faster or implement new features
Feel free to follow [the instructions](https://github.com/cowjen01/repsys#the-team) how to run RepSys locally (without docker) or contact [our Team](https://github.com/cowjen01/repsys#the-team).