import numpy as np

from util import is_dollar_amount, log_dollar_amount


def token_features(row, config):
    if row:
        c = config
        tokstr = row["token"].upper()
        return [
            (hash(tokstr) % config.vocab_size) if c.use_string else 0,
            float(row["page"]) if c.use_page else 0,
            float(row["x0"]) if c.use_geom else 0,
            float(row["y0"]) if c.use_geom else 0,
            float(len(tokstr)),
            fraction_digits(tokstr),
            is_dollar_amount(tokstr) or 0,
            log_dollar_amount(tokstr) or 0 if c.use_amount else 0,
        ]
    else:
        return [0 for i in range(config.token_dims)]


def fraction_digits(s):
    """Return the fraction of a string that is composed of digits."""
    return np.mean([c.isdigit() for c in s])


def base_features(token_df):
    """Extend a DataFrame with features that can be pre-computed."""
    token_df["hash"] = token_df["token"].apply(hash)
    token_df["length"] = token_df["token"].str.len()
    token_df["digitness"] = token_df["token"].apply(fraction_digits)
    token_df["is_dollar"] = token_df["token"].apply(is_dollar_amount)
    token_df["log_amount"] = token_df["token"].apply(log_dollar_amount)
    return token_df
