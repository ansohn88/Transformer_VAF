import pickle
from typing import Optional, Union

import numpy as np
import pandas as pd


def create_struct_nparr(
    events: np.ndarray,
    times: np.ndarray
) -> np.ndarray:
    dt = np.dtype(
        [('event', np.bool_), ('time', np.float_)]
    )
    arr = np.zeros(len(events), dt)
    arr['event'] = events
    arr['time'] = times
    return arr


def get_events_time(
        df: Union[str, pd.DataFrame],
        tcga_ids: list[str],
        which_time_metric: str = 'OS'
) -> dict:
    if isinstance(df, str):
        df = pd.read_pickle(df)

    sub_df = df.loc[
        df.bcr_patient_barcode.isin(tcga_ids)
    ]
    time_metric = f'{which_time_metric}.time'

    g = sub_df.groupby('bcr_patient_barcode')
    tids, events, times = list(
        zip(
            *[
                [
                    name,
                    np.unique(values['event'].values),
                    np.unique(values[time_metric].values)
                ]
                for name, values in g]
        )
    )
    events = np.asarray(events).reshape((len(events),))
    times = np.asarray(times).reshape((len(times),))

    arr = create_struct_nparr(events, times)

    return arr


def get_cancer_quantile_times(d: dict,
                              saveas: Optional[str]
                              ) -> Optional[dict]:
    blca = []
    cesc = []
    coad = []
    esca = []
    hnsc = []
    luad = []
    lusc = []
    ov = []
    paad = []
    sarc = []
    stad = []
    ucec = []
    for k in d:
        id = d[k]
        tb = id['time_bins']

        if id['type'] == 'BLCA':
            blca.append(tb)
        elif id['type'] == 'CESC':
            cesc.append(tb)
        elif id['type'] == 'COAD':
            coad.append(tb)
        elif id['type'] == 'ESCA':
            esca.append(tb)
        elif id['type'] == 'HNSC':
            hnsc.append(tb)
        elif id['type'] == 'LUAD':
            luad.append(tb)
        elif id['type'] == 'LUSC':
            lusc.append(tb)
        elif id['type'] == 'OV':
            ov.append(tb)
        elif id['type'] == 'PAAD':
            paad.append(tb)
        elif id['type'] == 'SARC':
            sarc.append(tb)
        elif id['type'] == 'STAD':
            stad.append(tb)
        elif id['type'] == 'UCEC':
            ucec.append(tb)

    ct_tb = {
        'BLCA': np.unique(np.asarray(blca), return_counts=True)[0],
        'CESC': np.unique(np.asarray(cesc), return_counts=True)[0],
        'COAD': np.unique(np.asarray(coad), return_counts=True)[0],
        'ESCA': np.unique(np.asarray(esca), return_counts=True)[0],
        'HNSC': np.unique(np.asarray(hnsc), return_counts=True)[0],
        'LUAD': np.unique(np.asarray(luad), return_counts=True)[0],
        'LUSC': np.unique(np.asarray(lusc), return_counts=True)[0],
        'OV': np.unique(np.asarray(ov), return_counts=True)[0],
        'PAAD': np.unique(np.asarray(paad), return_counts=True)[0],
        'SARC': np.unique(np.asarray(sarc), return_counts=True)[0],
        'STAD': np.unique(np.asarray(stad), return_counts=True)[0],
        'UCEC': np.unique(np.asarray(ucec), return_counts=True)[0]
    }

    if saveas is not None:
        with open(saveas, 'wb') as f:
            pickle.dump(ct_tb, f)
    else:
        return ct_tb
