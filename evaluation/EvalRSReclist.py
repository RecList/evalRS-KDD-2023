from reclist.logs import LOGGER
from reclist.metadata import METADATA_STORE
import pandas as pd
import numpy as np
import os
from reclist.reclist import rec_test
from reclist.reclist import RecList, CHART_TYPE
from random import choice

TOP_K_CHALLENGE = 100

class EvalRSReclist(RecList):

    def __init__(
        self,
        dataset,
        predictions,
        model_name,
        logger: LOGGER,
        metadata_store: METADATA_STORE,
        **kwargs
    ):
        super().__init__(
            model_name,
            logger,
            metadata_store,
            **kwargs
        )
        self.dataset = dataset
        self._x_train = dataset._get_train_set(fold=0)
        subset_of_cols_to_return = ['user_id', 'track_id']
        self._x_test = dataset._get_test_set(fold=0, subset_of_cols_to_return=subset_of_cols_to_return)[['user_id']]
        self._y_test = dataset._get_test_set(fold=0, subset_of_cols_to_return=subset_of_cols_to_return).set_index('user_id')
        self._y_preds = predictions
        self.similarity_model = kwargs.get("similarity_model", None)

        return
    
    def mrr_at_k_slice(self,
                        y_preds: pd.DataFrame,
                        y_test: pd.DataFrame,
                        slice_info: pd.DataFrame,
                        slice_key: str):

        from reclist.metrics.standard_metrics import rr_at_k
        # get rr (reciprocal rank) for each prediction made
        rr = rr_at_k(y_preds, y_test, k=TOP_K_CHALLENGE)
        # convert to DataFrame
        rr = pd.DataFrame(rr, columns=['rr'], index=y_test.index)
        # grab slice info
        rr[slice_key] = slice_info[slice_key].values
        # group-by slice and get per-slice mrr
        return rr.groupby(slice_key)['rr'].agg('mean').to_json()

    def miss_rate_at_k_slice(self,
                                   y_preds: pd.DataFrame,
                                   y_test: pd.DataFrame,
                                   slice_info: pd.DataFrame,
                                   slice_key: str):
        from reclist.metrics.standard_metrics import misses_at_k
        # get false positives
        m = misses_at_k(y_preds, y_test, k=TOP_K_CHALLENGE).min(axis=2)
        # convert to dataframe
        m = pd.DataFrame(m, columns=['mr'], index=y_test.index)
        # grab slice info
        m[slice_key] = slice_info[slice_key].values
        # group-by slice and get per-slice mrr
        return m.groupby(slice_key)['mr'].agg('mean')

    def miss_rate_equality_difference(self,
                                      y_preds: pd.DataFrame,
                                      y_test: pd.DataFrame,
                                      slice_info: pd.DataFrame,
                                      slice_key: str):
        from reclist.metrics.standard_metrics import misses_at_k

        mr_per_slice = self.miss_rate_at_k_slice(y_preds, y_test, slice_info, slice_key)
        mr = misses_at_k(y_preds, y_test, k=TOP_K_CHALLENGE).min(axis=2).mean()
        # take negation so that higher values => better fairness
        mred = -(mr_per_slice-mr).abs().mean()
        res = mr_per_slice.to_dict()
        return {'mred': mred, 'mr': mr, **res}

    def cosine_sim(self, u: np.array, v: np.array) -> np.array:
        return np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))

    @rec_test('stats')
    def stats(self):
        tracks_per_users = (self._y_test.values!=-1).sum(axis=1)
        return {
            'num_users': len(self._x_test['user_id'].unique()),
            'max_items': int(tracks_per_users.max()),
            'min_items': int(tracks_per_users.min())
        }

    @rec_test('HIT_RATE')
    def hit_rate_at_100(self):
        from reclist.metrics.standard_metrics import hit_rate_at_k
        hr = hit_rate_at_k(self._y_preds, self._y_test, k=TOP_K_CHALLENGE)
        return hr

    @rec_test('MRR')
    def mrr_at_100(self):
        from reclist.metrics.standard_metrics import mrr_at_k
        return mrr_at_k(self._y_preds, self._y_test, k=TOP_K_CHALLENGE)

    @rec_test('MRED_COUNTRY', display_type=CHART_TYPE.BARS)
    def mred_country(self):
        country_list = ["US", "RU", "DE", "UK", "PL", "BR", "FI", "NL", "ES", "SE", "UA", "CA", "FR", "NaN"]
        user_countries = self.dataset.df_users.loc[self._y_test.index, ['country']].fillna('NaN')
        valid_country_mask = user_countries['country'].isin(country_list)
        y_pred_valid = self._y_preds[valid_country_mask]
        y_test_valid = self._y_test[valid_country_mask]
        user_countries = user_countries[valid_country_mask]

        return self.miss_rate_equality_difference(y_pred_valid, y_test_valid, user_countries, 'country')

    @rec_test('MRED_USER_ACTIVITY', display_type=CHART_TYPE.BARS)
    def mred_user_activity(self):
        bins = np.array([1, 100, 1000])
        user_activity = self._x_train[self._x_train['user_id'].isin(self._y_test.index)]
        user_activity = user_activity.groupby('user_id',as_index=True, sort=False)[['user_track_count']].sum()
        user_activity = user_activity.loc[self._y_test.index]

        user_activity['bin_index'] = np.digitize(user_activity.values.reshape(-1), bins)
        user_activity['bins'] = bins[user_activity['bin_index'].values-1]

        return self.miss_rate_equality_difference(self._y_preds, self._y_test, user_activity, 'bins')

    @rec_test('MRED_TRACK_POPULARITY', display_type=CHART_TYPE.BARS)
    def mred_track_popularity(self):
        bins = np.array([1, 10, 100, 1000])
        track_id = self._y_test['track_id']
        track_activity = self._x_train[self._x_train['track_id'].isin(track_id)]
        track_activity = track_activity.groupby('track_id', as_index=True, sort=False)[['user_track_count']].sum()
        track_activity = track_activity.loc[track_id]

        track_activity['bin_index'] = np.digitize(track_activity.values.reshape(-1), bins)
        track_activity['bins'] = bins[track_activity['bin_index'].values - 1]

        return self.miss_rate_equality_difference(self._y_preds, self._y_test, track_activity, 'bins')

    @rec_test('MRED_ARTIST_POPULARITY', display_type=CHART_TYPE.BARS)
    def mred_artist_popularity(self):
        bins = np.array([1, 100, 1000, 10000])
        artist_id = self.dataset.df_tracks.loc[self._y_test['track_id'], 'artist_id']
        artist_activity = self._x_train[self._x_train['artist_id'].isin(artist_id)]
        artist_activity = artist_activity.groupby('artist_id', as_index=True, sort=False)[['user_track_count']].sum()
        artist_activity = artist_activity.loc[artist_id]

        artist_activity['bin_index'] = np.digitize(artist_activity.values.reshape(-1), bins)
        artist_activity['bins'] = bins[artist_activity['bin_index'].values - 1]

        return self.miss_rate_equality_difference(self._y_preds, self._y_test, artist_activity, 'bins')

    @rec_test('MRED_GENDER', display_type=CHART_TYPE.BARS)
    def mred_gender(self):
        user_gender = self.dataset.df_users.loc[self._y_test.index, ['gender']]
        return self.miss_rate_equality_difference(self._y_preds, self._y_test, user_gender, 'gender')

    @rec_test('BEING_LESS_WRONG')
    def being_less_wrong(self):
        from reclist.metrics.standard_metrics import hits_at_k

        hits = hits_at_k(self._y_preds, self._y_test, k=TOP_K_CHALLENGE).max(axis=2)
        misses = (hits == False)
        miss_gt_vectors = self.similarity_model[self._y_test.loc[misses, 'track_id'].values.reshape(-1)]
        # we calculate the score w.r.t to the first prediction
        miss_pred_vectors = self.similarity_model[self._y_preds.loc[misses, '0'].values.reshape(-1)]

        return float(self.cosine_sim(miss_gt_vectors, miss_pred_vectors).mean())
    

class EvalRSSimpleModel:
    """
    This is a dummy model that returns random predictions on the EvalRS dataset.
    """
    def __init__(self, items: pd.DataFrame, top_k: int=10, **kwargs):
        self.items = items
        self.top_k = top_k
        print("Received additional arguments: {}".format(kwargs))
        return

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        k = self.top_k
        num_users = len(user_ids)
        pred = self.items.sample(n=k*num_users, replace=True).index.values
        pred = pred.reshape(num_users, k)
        pred = np.concatenate((user_ids[['user_id']].values, pred), axis=1)
        return pd.DataFrame(pred, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')


