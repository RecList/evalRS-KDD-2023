"""

    This is the class containing the common methods for the evaluation of your model.

"""
import os
import time
import hashlib
import pandas as pd
from typing import Tuple
from utils import download_with_progress, get_cache_directory, LFM_DATASET_PATH, decompress_zipfile

class ChallengeDataset:

    def __init__(
        self,
        num_folds:int = 1,
        seed: int = None,
        force_download: bool = False
    ):
        
        # download dataset
        self.path_to_dataset = os.path.join(get_cache_directory(), 'evalrs_dataset_KDD_2023')
        if not os.path.exists(self.path_to_dataset) or force_download:
            print("Downloading LFM dataset...")
            download_with_progress(LFM_DATASET_PATH, os.path.join(get_cache_directory(), 'evalrs_dataset.zip'))
            decompress_zipfile(os.path.join(get_cache_directory(), 'evalrs_dataset.zip'),
                               get_cache_directory())
        else:
            print("LFM dataset already downloaded. Skipping download.")

        self.path_to_events = os.path.join(self.path_to_dataset, 'evalrs_events.parquet')
        self.path_to_tracks = os.path.join(self.path_to_dataset, 'evalrs_tracks.parquet')
        self.path_to_users = os.path.join(self.path_to_dataset, 'evalrs_users.parquet')
        self.path_to_topics = os.path.join(self.path_to_dataset, 'evalrs_topics.parquet')
        self.path_to_emotion_tags = os.path.join(self.path_to_dataset, 'evalrs_emotion_tags.parquet')
        self.path_to_social_tags = os.path.join(self.path_to_dataset, 'evalrs_social_tags.parquet')
        self.path_to_song_embeddings = os.path.join(self.path_to_dataset, 'evalrs_song_embeddings.parquet')

        assert os.path.exists(self.path_to_events)
        assert os.path.exists(self.path_to_tracks)
        assert os.path.exists(self.path_to_users)
        assert os.path.exists(self.path_to_topics)
        assert os.path.exists(self.path_to_emotion_tags)
        assert os.path.exists(self.path_to_social_tags)
        assert os.path.exists(self.path_to_song_embeddings)

        print("Loading dataset.")
        self.df_events = pd.read_parquet(self.path_to_events)
        self.df_tracks = pd.read_parquet(self.path_to_tracks).set_index('track_id')
        self.df_users = pd.read_parquet(self.path_to_users).set_index('user_id')
        self.df_topics = pd.read_parquet(self.path_to_topics).set_index('urlSong')
        self.df_emotion_tags = pd.read_parquet(self.path_to_emotion_tags).set_index('lastfm_id')
        self.df_social_tags = pd.read_parquet(self.path_to_social_tags).set_index('lastfm_id')
        self.df_song_embeddings = pd.read_parquet(self.path_to_song_embeddings).set_index('urlSong')

        print("Generating Train/Test Split.")
        self.num_folds = num_folds
        self.unique_user_ids_df = self.df_events[['user_id']].drop_duplicates()
        self._random_state = int(time.time()) if not seed else seed
        self._train_set, self._test_set = self._generate_folds(self.num_folds, self._random_state)

        print("Generating dataset hashes.")
        self._events_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df_events.sample(n=1000,
                                                                                            random_state=0)).values
                                           ).hexdigest()
        self._tracks_hash = hashlib.sha256(pd.util.hash_pandas_object
                                           (self.df_tracks.sample(n=1000, random_state=0)
                                            .explode(['albums', 'albums_id'])).values
                                           ).hexdigest()
        self._users_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df_users.sample(n=1000,
                                                                                          random_state=0)).values
                                          ).hexdigest()


    def _get_vertice_count(self, df, column_name:str):
        df = df.copy()
        df['counts'] = 1
        counts = df.groupby(column_name, as_index=True)[['counts']].sum()
        return counts

    def _k_core_filter(self, df_user_track: pd.DataFrame, k=10):
        num_users_prev, num_tracks_prev = None, None
        delta = True
        iter, max_iter = 0, 10
        valid_users = df_user_track['user_id'].unique()
        valid_tracks = df_user_track['track_id'].unique()
        while delta and iter < max_iter:
            track_counts = self._get_vertice_count(df_user_track, 'track_id')
            valid_tracks = track_counts[track_counts['counts']>=k].index
            # keep only valid tracks
            df_user_track = df_user_track[df_user_track['track_id'].isin(valid_tracks)]
            user_counts = self._get_vertice_count(df_user_track, 'user_id')
            valid_users = user_counts[user_counts['counts']>=k].index
            # keep only valid users
            df_user_track = df_user_track[df_user_track['user_id'].isin(valid_users)]

            num_tracks = len(valid_tracks)
            num_users = len(valid_users)
            # check for any update
            delta = (num_users != num_users_prev) or (num_tracks != num_tracks_prev)

            num_users_prev = num_users
            num_tracks_prev = num_tracks
            iter+=1

            # # DEBUG
            # print("ITER {}".format(iter))
            # print("THERE ARE {} VALID TRACKS; MIN VERTICES {}".format(len(valid_tracks), track_counts['counts'].min()))
            # print("THERE ARE {} VALID USERS; MIN VERTICES {}".format(len(valid_users), user_counts['counts'].min()))

        return valid_users, valid_tracks

    def _generate_folds(self, num_folds: int, seed: int, frac=0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:

        fold_ids = [(self.unique_user_ids_df.sample(frac=frac, random_state=seed+_)
                     .reset_index(drop=True)
                     .rename({'user_id': _}, axis=1)) for _ in range(num_folds)]
        # in theory all users should have at least 10 interactions
        df_fold_user_ids = pd.concat(fold_ids, axis=1)

        test_dfs = []
        train_dfs_idx = []
        for fold in range(num_folds):
            df_fold_events = self.df_events[self.df_events['user_id'].isin(df_fold_user_ids[fold])]
            # perform k-core filter; threshold of 10
            valid_user_ids, valid_track_ids = self._k_core_filter(df_fold_events[['user_id','track_id']], k=10)

            df_fold_events = df_fold_events[df_fold_events['user_id'].isin(valid_user_ids)]
            df_fold_events = df_fold_events[df_fold_events['track_id'].isin(valid_track_ids)]

            df_groupby = df_fold_events.groupby(by='user_id', as_index=False)

            subset_of_cols_to_return = ['user_id', 'track_id']

            df_test = df_groupby.sample(n=1, random_state=seed)[subset_of_cols_to_return]
            df_test['fold'] = fold
            df_train = df_fold_events.index.difference(df_test.index).to_frame(name='index')
            df_train['fold'] = fold

            test_dfs.append(df_test)
            train_dfs_idx.append(df_train)
            # unique_user_id = pd.DataFrame(df_fold_events['user_id'].unique(), columns=['user_id'])
            # unique_user_id['fold'] = fold
            # fold_user_ids.append(unique_user_id)

        df_test = pd.concat(test_dfs, axis=0)
        df_train = pd.concat(train_dfs_idx, axis=0)

        # print(df_test)
        # print('====')
        # print(df_users)
        # print('====')
        return df_train, df_test

    def _get_train_set(self, fold: int) -> pd.DataFrame:
        assert fold <= self._test_set['fold'].max()
        train_index =self._train_set[self._train_set['fold']==fold]['index']
        # test_index = self._test_set[self._test_set['fold']==fold].index
        # fold_users = self._fold_ids[self._fold_ids['fold']==fold]['user_id']
        # train_fold = (self.df_events.loc[self.df_events['user_id'].isin(fold_users)])

        return self.df_events.loc[train_index]

    def _get_test_set(self, fold: int, limit: int = None, seed: int =0) -> pd.DataFrame:
        subset_of_cols_to_return = ['user_id', 'track_id']

        assert fold <= self._test_set['fold'].max()
        test_set = self._test_set[self._test_set['fold'] == fold][subset_of_cols_to_return]
        if limit:
            return test_set.sample(n=limit, random_state=seed)
        else:
            return test_set

    def get_sample_train_test(self):
        return self._get_train_set(0), self._get_test_set(0)




if __name__ == "__main__":
    dataset = ChallengeDataset(force_download=False)
    print(dataset._get_train_set(0))
    print(dataset._get_test_set(0))
    print(dataset.df_tracks)

    # print(dataset.df_events.head())
    # print(dataset.df_tracks.head())
    # print(dataset.df_users.head())
    # print(dataset.df_topics.head())
    # print(dataset.df_emotion_tags.head())
    # print(dataset.df_social_tags.head())
    # print(dataset.df_song_embeddings.head())
