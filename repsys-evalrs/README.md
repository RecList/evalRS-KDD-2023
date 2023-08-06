## Implementation of [RepSys](https://github.com/cowjen01/repsys/) for EvalRS 2023 workshop

### I just want to try RepSys without editing the dataset and models:

A live version of RepSys is available at [http://mzai-repsys.tenant-d43f85-prod.coreweave.cloud:3001/](http://mzai-repsys.tenant-d43f85-prod.coreweave.cloud:3001/). RepSys can also be run locally on any device with at least 8GB of RAM. We have a docker image that contains the implementation of the EvalRS 2023 dataset along with the parquet files and trained models:
```
docker run --rm --name repsys -p 3001:3001 kasape/repsys:kdd-with-data
```

### I want to play with the EvalRS dataset using RepSys and alter the dataset or define a new model and visualize it in RepSys (python package version):

RepSys can be installed from [pypi](https://pypi.org/project/repsys-framework/). Supported python versions are 3.8, 3.9, 3.10, 3.11. We can run RepSys for EvalRS dataset by installing RepSys from pypi along with other python packages used by our dataset and models.
**P1. Clone the main workshop repository containing implementation of EvalRS dataset for Repsys in `repsys-evalrs` folder:**
```
git clone git@github.com:RecList/evalRS-KDD-2023.git
cd evalRS-KDD-2023/repsys-evalrs/
```
**P2. Download the dataset EvalRS KDD (1.60GB) and unzip it:**
```
wget https://evalrs.object.lga1.coreweave.com/evalrs_dataset_KDD_2023.zip
unzip evalrs_dataset_KDD_2023.zip 
```
**P3. The implementation of the EvalRS dataset for RepSys expects the data to be found in the `data` folder.**
```
mv evalrs_dataset_KDD_2023 data
```
*Note: For simplicity, we move all the files, but in fact, RepSys uses only `evalrs_emotion_tags.parquet.parquet`, `evalrs_events.parquet`, `evalrs_social_tags.parquet` and `evalrs_tracks.parquet`.*

**P4. Install repsys and other packages, preferably to virtual environment:**
```
# python -m venv venv 
# source venv/bin/activate
pip install -r requirements.txt
```
Once *repsys-framework* package is installed, RepSys CLI is available. However, before we can use RepSys to vizualize the data, we need to call RepSys CLI to: 
1. Split the dataset into training and validation parts using the following command. A pruning specified in the `repsys.ini` is applied, as described [in the documentation](https://github.com/cowjen01/repsys/tree/master#repsysini). As a result of this step, folder `.repsys_checkpoints` is created (if not existed before) and it contains `dataset-split-[timestamp].zip` file. 
```
repsys dataset split
```

2. Train all models in `models/*.py` files or specify a model or specify model by appending `-m [model_name]` to the following command:

```
repsys model train
```

3. Evalute all models (or specify one using `-m [model_name]`) producing files `.repsys_checkpoints/model-eval-[model-name]-[timestamp].zip`
```
repsys model eval
```

4. Evaluate the dataset (by producing a file `.repsys_checkpoints/dataset-eval-[model-name]-[timestamp].zip`) with optionally specified method as described in [the documentation](https://github.com/cowjen01/repsys/#evaluating-the-dataset)
```
repsys dataset eval
```

**P5. Check we have the following files in `repsys-evalrs/.repsys_checkpoints` folder:**
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
**P6. Run RepSys server with command:**
```
repsys server
```

### I want to play with the EvalRS dataset using RepSys and alter the dataset or define a new model and visualize it in RepSys (Docker version, not-optimized for ARM architecture of new Macbooks):

RepSys can be run as a docker container with the implementation of the EvalRS 2023 dataset and its data mounted to a folder `/app` by adding `-v $HOME/<path-to-repsys-evalrs>:/app`. To run RepSys in a docker container, we can follow the steps explaining how to run RepSys installed from pypi with the modification that RepSys CLI is called as an entrypoint. Namely, we perform steps **P1**, **P2**, **P3** to download data and implementation of EvalRS dataset. Step *P4* is different because we will call RepSys CLI in the docker container instead of calling it directly:
```
docker run --rm -v $HOME/<path-to-repsys-evalrs>:/app kasape/repsys:kdd dataset split
docker run --rm -v $HOME/<path-to-repsys-evalrs>:/app kasape/repsys:kdd model train
docker run --rm -v $HOME/<path-to-repsys-evalrs>:/app kasape/repsys:kdd model eval
docker run --rm -v $HOME/<path-to-repsys-evalrs>:/app kasape/repsys:kdd dataset eval
```
As a result of the previous steps, our `$HOME/<path-to-repsys-evalrs>` folder should contain all files listed in **P5**. Repsys can be run as a docker container with:**
```
docker run --rm -v $HOME/<path-to-repsys-evalrs>:/app -p 3001:3001 kasape/repsys:kdd
```

#### Troubleshooting

###### Some of the commands `dataset split`, `model train`, `model eval`, `dataset eval` are taking forever.

For model or dataset evaluation, you can set higher pruning (`min_item_interacts` and `min_user_interacts`) to keep only items and users that are really important in the dataset to speed up the computation. After editing pruning, the new split needs to be created (`dataset split`). For model training, you can configure models to be smaller, e.g. alter `repsys-evalrs/models/elsa.py` to have a smaller latent space size (`n_dims=64`) or a smaller number of epochs.

###### I have edited `repsys-evalrs/dataset.py` to have different items columns or load extra information in `load_items` method, but when I run RecSys, nothing changes.

RecSys needs to preprocess the dataset into its own format and map all the columns of items. Please call again `repsys dataset split`.

###### I want to add custom python package to `kasape/repsys:kdd` docker image and use it in dataset loading or in my own model.

You can edit `repsys-evalrs/requirements.txt` and build your version of `kasape/repsys:kdd` using provided `repsys-evalrs/Dockerfile` by calling `docker build -t kasape/repsys:kdd-mine -f Dockerfile .` inside the `repsys-evalrs` folder.

###### I don't have docker, but I want to try RepSys.

RepSys is also possible to use as a python package installed from pypi. Please see the [official RepSys repository](https://github.com/cowjen01/repsys/) for more instructions.

###### I've setup PyMDE as `embed_method` in `repsys.ini` or I've called `dataset evaluate --method pymde` and I'm getting ` ModuleNotFoundError: No module named 'pymde'`.

PyMDE is not included in the original list of requirements.txt for RepSys, because it was making the docker image unnecessarily large. If you want to use it, you can add `pymde==0.1.14` as specified in **I want to add custom python package**, or follow the [instructions](https://github.com/cowjen01/repsys/#installation) if you are installing RepSys from pypi. 

###### I'm getting `Invalid value for '-c' / '--config': Path 'repsys.ini' does not exist`

In the case of the docker version, folder *repsys-evalrs* wasn't properly mounted to the docker container and RepSys framework cannot find `/app/repsys.ini` file. For RepSys installed from pypi, it is necessary to call Repsys CLI from the folder containing `repsys.ini`, `dataset.py` and other files.

### I want to modify the core of RepSys to make it faster or implement new features
Feel free to follow [the instructions](https://github.com/cowjen01/repsys#contributing) how to run RepSys locally or contact [our Team](https://github.com/cowjen01/repsys#the-team).