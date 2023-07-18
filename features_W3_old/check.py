'''
A Data Leakage Check Function.

Assumption:

- Data Leakage means using future data to calculate feature data.
- The 'Feature Extraction Function' will not reset the dataframe index.

Method:

1. Call the 'Feature Extraction Function' to the original dataset.
2. Create 'test dataset' which removed some rows at last. (i.e., df[:-100])
3. Call the 'Feature Extraction Function' to the 'test dataset'.
4. Align both dataset with the dataframe index.
5. Check any value difference from both dataset.
    - If values are different -> Data Leakage!
    - If values are the same -> No Leakage, Bravo.
'''

import pandas as pd
import numpy as np
from typing import  Callable, Dict, Any

def check_leakage(df: pd.DataFrame, extractor: Callable[..., pd.DataFrame], extractor_args: Dict, df_field='df', remove_rows=100):
    config_a = extractor_args.copy()
    config_a[df_field] = df.copy()
    config_b = extractor_args.copy()
    config_b[df_field] = df.copy()[:-remove_rows]

    test_a = extractor(**config_a)
    test_b = extractor(**config_b)

    # Align both dataframe
    test_a = test_a[test_a.index.isin(test_b.index)]
    test_b = test_b[test_b.index.isin(test_a.index)]
    for index_a, index_b in zip(test_a.index, test_b.index):
        assert index_a == index_b

    # Check their values (tolerance for floating point precision)
    result = np.allclose(test_a.values, test_b.values, atol=1e-6)
    return result