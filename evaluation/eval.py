import os
from EvalRSRunner import ChallengeDataset
from EvalRSReclist import EvalRSSimpleModel, EvalRSReclist
from reclist.reclist import LOGGER, METADATA_STORE
from gensim.models import KeyedVectors


if __name__ == '__main__':

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
        logger=LOGGER.LOCAL,
        metadata_store=METADATA_STORE.LOCAL,
        similarity_model=similarity_model,
    )
    
    # run reclist
    cdf(verbose=True)
