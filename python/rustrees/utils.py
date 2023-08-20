import pandas as pd
import pyarrow as pa
import rustrees.rustrees as rt


def from_pandas(df: pd.DataFrame) -> rt.Dataset:
    """
    Convert a Pandas DataFrame to a Rustrees Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.

    Returns
    -------
    rt.Dataset
        The Rustrees Dataset.
    """
    record_batch = pa.RecordBatch.from_pandas(df)
    return rt.Dataset.from_pyarrow(record_batch)


def prepare_dataset(X, y=None) -> rt.Dataset:
    """
    Prepare a Rustrees Dataset from a Pandas DataFrame or a 2D array-like object.

    Parameters
    ----------
    X : pd.DataFrame or 2D array-like object
        The features.
    y : list, Numpy array, or Pandas Series, optional
        The target. The default is None.

    Returns
    -------
    rt.Dataset
        The Rustrees Dataset.

    Raises
    ------
    ValueError
        If X is not a Pandas DataFrame or a 2D array-like object.
        If y is not a list, Numpy array, or Pandas Series.
    """
    if isinstance(X, pd.DataFrame):
        dataset = from_pandas(X)
    else:
        try:
            dataset = from_pandas(pd.DataFrame(X))
        except Exception as e:
            raise ValueError(
                "X must be a Pandas DataFrame or a 2D array-like object"
                + "that you can call `pd.DataFrame(X)` on."
            )
    if y is not None:
        try:
            dataset.add_target(y)
        except Exception as e:
            raise ValueError("y must be a list, Numpy array, or a Pandas Series.")
    return dataset
