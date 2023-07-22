from EvalRSRunner import ChallengeDataset
from EvalRSReclist import EvalRSSimpleModel, EvalRSReclist
from reclist.reclist import LOGGER, METADATA_STORE


if __name__ == '__main__':

    # load dataset
    dataset = ChallengeDataset()

    # dummy model
    my_df_model = EvalRSSimpleModel(dataset.df_tracks, top_k=100)
    # get some predictions
    df_predictions = my_df_model.predict(dataset._get_test_set(fold=0)[['user_id']])
    
    # initialize with everything
    cdf = EvalRSReclist(
        dataset=dataset,
        model_name="SimpleModel",
        predictions=df_predictions,
        logger=LOGGER.LOCAL,
        metadata_store=METADATA_STORE.LOCAL,
    )
    
    # run reclist
    cdf(verbose=True)
