# Tutorial for the EvalRS 2023 - RecSys evaluation hackaton

## Retrieval models with Merlin and Tensorflow on LastFM dataset


In this folder we provide a baseline recommender algorithms using the [Merlin framework](https://github.com/NVIDIA-Merlin/) and Tensorlfow.
It builds, trains and evaluates two retrieval models: **Matrix Factorization** and **Two-Tower architecture**.  

### Setup
For easier setup we recommend using Docker.

1. Run the following command pulls and run the Merlin TensorFlow Container (23.06 release).  
Please set the host path to the folder where you have pulled the `evalRS-KDD-2023` repo from GitHub and also the path where the dataset will be downloaded and preprocessed.

```bash
docker run --runtime=nvidia --rm -it --ipc=host --cap-add SYS_NICE -v /PATH/TO/evalRS-KDD-2023:/evalRS-KDD-2023 -v /PATH/TO/DATASET/WORKSPACE:/data -p 8888:8888 nvcr.io/nvidia/merlin/merlin-tensorflow:23.06 /bin/bash
```

2. Inside the Docker container, pull the [RecList](https://github.com/RecList/reclist) library and install it. RecList will be used for evaluation.

```bash
mkdir -p /workspace/ && cd /workspace/
git clone https://github.com/Reclist/reclist/
cd reclist && pip install -e .
```

3. Also inside the Docker container, update the [Models](https://github.com/NVIDIA-Merlin/models/) library to the [evalRS_2023](https://github.com/NVIDIA-Merlin/models/tree/evalrs_2023), which has some fixes necessary for pre-trained embeddings support.

```bash
cd /models
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && git fetch origin evalRS_2023 && git checkout evalRS_2023
pip install . --no-deps
```

4. Start Jupyter notebook inside the container and run the [Merlin tutorial notebook](evalrs_kdd_2023_tutorial_retrieval_models_with_merlin_tf.ipynb).

```bash
cd /evalRS-KDD-2023/notebooks/merlin_tf_tutorial
jupyter notebook --no-browser --ip 0.0.0.0 --no-browser --allow-root
```