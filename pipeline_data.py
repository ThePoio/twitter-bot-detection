"""Data processing pipeline for twitter-bot-detection dataset.

This module provides a `process_df` function that adds the following features:

- `account_age_days`: integer days between `Created At` and reference date (default: now)
- `username_digits`: count of numeric characters in `Username`
- `has_hashtags`: 1 if `Hashtags` contains any non-empty value, else 0
- `tweet_length`: number of characters in `Tweet`
- `url_count`: number of URLs found in `Tweet`

Usage example:
    df = pd.read_csv('.data/bot_detection_data.csv')
    df = process_df(df)
    df.to_csv('.data/bot_detection_data_processed.csv', index=False)
"""

import re
from datetime import datetime
import pandas as pd
import numpy as np


URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce')


def account_age_days(created_at: pd.Series, reference: datetime | None = None) -> pd.Series:
    """Return account age in days (integer). NaT -> NaN."""
    ref = reference or datetime.utcnow()
    created = _parse_datetime(created_at)
    delta = ref - created
    # where created is NaT, result should be NaN
    return delta.dt.days.astype('float')


def username_digits(username: pd.Series) -> pd.Series:
    """Count digit characters in the username."""
    return username.fillna('').astype(str).apply(lambda s: sum(c.isdigit() for c in s)).astype(int)


def has_hashtags(hashtags: pd.Series) -> pd.Series:
    """Return 1 if hashtags cell has at least one non-empty token, else 0."""
    def _has(x):
        if pd.isna(x):
            return 0
        s = str(x).strip()
        if s == '':
            return 0
        return 1

    return hashtags.apply(_has).astype(int)


def tweet_length(tweet: pd.Series) -> pd.Series:
    return tweet.fillna('').astype(str).apply(len).astype(int)


def url_count(tweet: pd.Series) -> pd.Series:
    return tweet.fillna('').astype(str).apply(lambda s: len(URL_REGEX.findall(s))).astype(int)


def process_df(df: pd.DataFrame, reference: datetime | None = None) -> pd.DataFrame:
    """Add derived features to the dataframe and return a new DataFrame.

    Does not modify the input `df` in-place (returns a copy).
    """
    out = df.copy()

    if 'Created At' in out.columns:
        out['Created At'] = _parse_datetime(out['Created At'])
        out['account_age_days'] = account_age_days(out['Created At'], reference=reference)
    else:
        out['account_age_days'] = np.nan

    # Username -> username_digits
    if 'Username' in out.columns:
        out['username_digits'] = username_digits(out['Username'])
    else:
        out['username_digits'] = 0

    # Hashtags -> has_hashtags
    if 'Hashtags' in out.columns:
        out['has_hashtags'] = has_hashtags(out['Hashtags'])
    else:
        out['has_hashtags'] = 0

    # Tweet -> tweet_length, url_count
    if 'Tweet' in out.columns:
        out['tweet_length'] = tweet_length(out['Tweet'])
        out['url_count'] = url_count(out['Tweet'])
    else:
        out['tweet_length'] = 0
        out['url_count'] = 0

    return out


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Process bot detection dataset and add derived features')
    p.add_argument('--input', '-i', default='.data/bot_detection_data.csv')
    p.add_argument('--output', '-o', default='.data/bot_detection_data_processed.csv')
    args = p.parse_args()

    df_in = pd.read_csv(args.input)
    df_out = process_df(df_in)
    df_out.to_csv(args.output, index=False)
    print(f'Processed {len(df_in)} rows -> {args.output}')
