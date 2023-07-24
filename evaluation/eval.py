import os
from EvalRSRunner import ChallengeDataset
from EvalRSReclist import EvalRSSimpleModel, EvalRSReclist
from reclist.reclist import LOGGER, METADATA_STORE
from gensim.models import KeyedVectors


if __name__ == '__main__':
    # support some trivial command-line arguments to leverage
    # reclist support for third-party experiment services
    import sys
    # add support for .env files to define env variables for loggers
    from dotenv import load_dotenv
    load_dotenv()

    my_logger = LOGGER.LOCAL
    my_logger_envs = {}
    if len(sys.argv) == 2:
        if sys.argv[1].lower()  == 'comet':
            my_logger = LOGGER.COMET
            # make sure we have the right envs to have the logger working
            assert os.environ["COMET_KEY"]
            assert os.environ["COMET_PROJECT_NAME"]
            assert os.environ["COMET_WORKSPACE"]
        elif sys.argv[1].lower()  == 'neptune':
            my_logger = LOGGER.NEPTUNE
            # make sure we have the right envs to have the logger working
            assert os.environ["NEPTUNE_KEY"]
            assert os.environ["NEPTUNE_PROJECT_NAME"]
        else:
            raise ValueError(f'Unknown logger: {sys.argv[1]}')

    # load dataset
    dataset = ChallengeDataset(force_download=False)

    # dummy model
    my_df_model = EvalRSSimpleModel(dataset.df_tracks, top_k=100)
    # get some predictions
    df_predictions = my_df_model.predict(dataset._get_test_set(fold=0)[['user_id']])

    # load a similarity model: here we used 
    similarity_model = KeyedVectors.load(os.path.join(dataset.path_to_dataset, 'song2vec.wv'))
    
    # initialize with everything
    cdf = EvalRSReclist(
        dataset=dataset,
        model_name="SimpleModel",
        predictions=df_predictions,
        logger=my_logger,
        metadata_store=METADATA_STORE.LOCAL,
        similarity_model=similarity_model
    )
    
    # run reclist
    cdf(verbose=True)
