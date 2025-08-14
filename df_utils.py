import pandas as pd
import numpy as np
def fill_zones(df, index_col='md', newIndexSeries = None):
    df.to_csv('./data/ori.csv')
    df_indexed = df.set_index(index_col)
    newIndex = df_indexed.index.union(newIndexSeries)
    df_indexed = df_indexed.reindex(newIndex)
    df_indexed['well'] = df_indexed['well'].ffill()
    df_indexed['PERF'] = 1

    sStopF = df_indexed.stop.ffill()
    sStartF = df_indexed.start.ffill()
    sStartB = df_indexed.start.bfill()
    sDiff = sStartF - sStartB
    sMasked = sDiff != 0
    
    df_indexed['start'] = sStartF.mask(sMasked, np.nan)
    df_indexed['stop'] = sStopF.mask(sMasked, np.nan)
    df_indexed['PERF'] = df_indexed.PERF.mask(sMasked, np.nan)
    df_indexed = df_indexed.reindex(newIndexSeries)
    df_indexed.to_csv('./data/interpolated.csv')
    df_indexed = df_indexed.reset_index()
    df_indexed.to_csv('./data/interpolated_reset.csv')
    print("=============")
    print(df_indexed[df_indexed.start.notnull()])
    return df_indexed
