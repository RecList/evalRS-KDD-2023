"""

    This is the class containing the common methods for the evaluation of your model.

"""
import os
import hashlib
import pandas as pd
from utils import download_with_progress, get_cache_directory, LFM_DATASET_PATH, decompress_zipfile

class ChallengeDataset:

    def __init__(self, num_folds=4, seed: int = None, force_download: bool = False):
        # download dataset
        self.path_to_dataset = os.path.join(get_cache_directory(), 'evalrs_dataset_KDD23')
        if not os.path.exists(self.path_to_dataset) or force_download:
            print("Downloading LFM dataset...")
            download_with_progress(LFM_DATASET_PATH, os.path.join(get_cache_directory(), 'evalrs_dataset.zip'))
            decompress_zipfile(os.path.join(get_cache_directory(), 'evalrs_dataset.zip'),
                               get_cache_directory())
        else:
            print("LFM dataset already downloaded. Skipping download.")

        self.path_to_events = os.path.join(self.path_to_dataset, 'evalrs_events.csv')
        self.path_to_tracks = os.path.join(self.path_to_dataset, 'evalrs_tracks.csv')
        self.path_to_users = os.path.join(self.path_to_dataset, 'evalrs_users.csv')
        self.path_to_topics = os.path.join(self.path_to_dataset, 'evalrs_topics.csv')
        self.path_to_emotion_tags = os.path.join(self.path_to_dataset, 'evalrs_emotion_tags.csv')
        self.path_to_social_tags = os.path.join(self.path_to_dataset, 'evalrs_social_tags.csv')

        assert os.path.exists(self.path_to_events)
        assert os.path.exists(self.path_to_tracks)
        assert os.path.exists(self.path_to_users)
        assert os.path.exists(self.path_to_topics)
        assert os.path.exists(self.path_to_emotion_tags)
        assert os.path.exists(self.path_to_social_tags)

        print("Loading dataset.")
        self.df_events = pd.read_csv(self.path_to_events, index_col=0, dtype='int32')
        self.df_tracks = pd.read_csv(self.path_to_tracks,
                                     dtype={
                                         'track_id': 'int32',
                                         'artist_id': 'int32'
                                     }).set_index('track_id')

        self.df_users = pd.read_csv(self.path_to_users,
                                    dtype={
                                        'user_id': 'int32',
                                        'playcount': 'int32',
                                        'country_id': 'int32',
                                        'timestamp': 'int32',
                                        'age': 'int32',
                                    }).set_index('user_id')

        self.df_topics = pd.read_csv(self.path_to_topics).set_index('urlSong')

        self.df_emotion_tags = pd.read_csv(self.path_to_emotion_tags).set_index('lastfm_id')

        self.df_social_tags = pd.read_csv(self.path_to_social_tags).set_index('lastfm_id')


if __name__ == "__main__":
    dataset = ChallengeDataset(force_download=True)

    print(dataset.df_events[:5])
    print(dataset.df_tracks[:5])
    print(dataset.df_users[:5])
    print(dataset.df_topics[:5])
    print(dataset.df_emotion_tags[:5])
    print(dataset.df_social_tags[:5])
