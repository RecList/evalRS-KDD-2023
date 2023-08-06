import ast

import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class EvalRS(Dataset):
    TAG_SEPARATOR = ","

    def name(self):
        return "evalrs"

    def item_cols(self):
        return {
            "track_id": dtypes.ItemID(),
            "track": dtypes.Title(),
            "artist": dtypes.String(),
            "albums_description": dtypes.String(),
            "emotion_tags": dtypes.Tag(sep=self.TAG_SEPARATOR),
            "social_tags": dtypes.Tag(sep=self.TAG_SEPARATOR),
        }

    def interaction_cols(self):
        return {"track_id": dtypes.ItemID(), "user_id": dtypes.UserID()}

    @staticmethod
    def __parse_albums(albums: pd.Series):
        def remove_nan(x):
            # Some list contains 'Hail to the Thief', nan, nan, 'Radiohead...' and we don't want to simply replace 'nan' with '' because 'nan' can also be part of the correct name
            # We need to check that 'nan' is without quotes and between commas (or other cases like on the beginning)
            # Repeated 'replace' is needed because patterns ', nan,' are overlapping so they won't get matched twice for more consecutive nan values
            while True:
                before = x
                # Replace nan on the beginning
                x = x.replace("[nan, ", "[")
                # Replace nan in the middle
                x = x.replace(",nan,", ",")
                x = x.replace(", nan,", ",")
                # Replace nan in the end
                x = x.replace(", nan]", "]")
                x = x.replace(",nan]", "]")
                # Replace nan when it's the only element
                x = x.replace("[nan]", '""')
                if before == x:
                    break
            return x

        albums_decoded = albums.apply(
            lambda x: list(set(ast.literal_eval(remove_nan(x))))
        )
        return albums_decoded

    @staticmethod
    def __join_with_nan(x: list):
        vals = [str(i) for i in x if i is not None and len(i) > 0]
        if len(vals) == 0:
            return "N/A"
        else:
            return EvalRS.TAG_SEPARATOR.join(vals)

    def load_items(self):
        tracks = pd.read_parquet("./data/evalrs_tracks.parquet")[
            ["track_id", "track", "albums", "artist", "lastfm_id"]
        ]
        emotion_tags = pd.read_parquet("./data/evalrs_emotion_tags.parquet")
        social_tags = pd.read_parquet("./data/evalrs_social_tags.parquet")

        albums = self.__parse_albums(tracks["albums"])
        # unique_albums = set([album for sublist in albums.tolist() for album in sublist])
        # There are 754521 unique album names, that's too much to use them as tags for vizualization.
        # We will case convert the list of album names to one short string as description
        tracks["albums_description"] = albums.apply(
            lambda x: f"Albums: {x[0]}, {x[1]} and {len(x) - 2} others"
            if len(x) > 3
            else "Albums: " + ", ".join(x)
        )
        # For recommendation based on the album, it's possible to train a token model (leave it to the hackaton)

        # There are only 197 unique emotion_tags (not including their weights) so we use them as tags for vizualization
        tracks_with_tags = tracks.merge(
            emotion_tags.groupby("lastfm_id")["emotion_tags_value"]
            .apply(lambda x: self.__join_with_nan(x))
            .reset_index(),
            on="lastfm_id",
            how="left",
        )
        # There are 387380 unique social tags, that's too much for vizualization in RepSys.
        # Tag are provided for 119330 different lastfm_id values (tracks).
        # If we take only the most popular 200 social tags, we cover 114628 (96%) of all tracks and we can visualize them:
        selected_social_tags = (
            social_tags.groupby("social_tags_value")
            .size()
            .sort_values(ascending=False)
            .head(200)
        )
        in_selected = social_tags["social_tags_value"].isin(selected_social_tags.index)
        tracks_with_tags = tracks_with_tags.merge(
            social_tags[in_selected]
            .groupby("lastfm_id")["social_tags_value"]
            .apply(lambda x: self.__join_with_nan(x))
            .reset_index(),
            on="lastfm_id",
            how="left",
        )
        tracks_with_tags.rename(
            columns={
                "emotion_tags_value": "emotion_tags",
                "social_tags_value": "social_tags",
            },
            inplace=True,
        )
        return tracks_with_tags

    def load_interactions(self):
        df = pd.read_parquet("./data/evalrs_events.parquet")
        return df[["user_id", "track_id"]]

    def web_default_config(self):
        return {
            "mappings": {
                "title": "track",
                "subtitle": "artist",
                "caption": "social_tags",
                "content": "albums_description",
            },
            "recommenders": [
                {
                    "name": "Random summer songs",
                    "model": "rand",
                    "modelParams": {"social_tags": "summer"},
                },
                {
                    "name": "Heartbreaking Rock by KNN",
                    "model": "knn",
                    "modelParams": {
                        "social_tags": "rock",
                        "emotion_tags": "heartbreaking",
                    },
                },
                {
                    "name": "The most popular funeral songs",
                    "model": "pop",
                    "modelParams": {"emotion_tags": "funeral"},
                },
                {"name": "Selected by ELSA", "model": "elsa"},
            ],
        }
