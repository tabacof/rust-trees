import pandas as pd
import rustrees.rustrees as rt
import pyarrow as pa

def from_pandas(df: pd.DataFrame) -> rt.Dataset:
    record_batch = pa.RecordBatch.from_pandas(df)
    return rt.Dataset.from_pyarrow(record_batch)

