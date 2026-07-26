import pandas as pd
from datasets import DatasetDict


def _as_dataframe(data: DatasetDict | pd.DataFrame) -> pd.DataFrame:
    """
    Generate a dataframe to run a test with from a dataset dict.

    :param data: The data to convert to a dataframe.
    :return: The dataframe representing the data. One row per item.
    """
    if isinstance(data, pd.DataFrame):
        return data

    list_frames = []
    for str_split_label, obj_dataset in data.items():
        str_split, str_scenario, str_principal = str_split_label.split("_", 2)
        df_split = obj_dataset.with_format(None).to_pandas()
        df_split["split"] = str_split
        df_split["scenario"] = str_scenario
        df_split["entity"] = str_principal
        df_split["category"] = "entity_swap"
        list_frames.append(df_split)
    return pd.concat(list_frames, ignore_index=True)
