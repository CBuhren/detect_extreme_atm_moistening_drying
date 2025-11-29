import numpy as np
import pandas as pd
import xarray as xr
import glob

def load_df_events_csv(filepath_event: str) -> pd.DataFrame:
    """
    Import the events csv file from given `filepath` and load as one pd.DataFrame

    Parameters:
    -----------
        - filepath: filepath to events (`str`)

    Returns:
    --------
        - df_events: all detected moistening and drying events (`pd.DataFrame`)
    """

    df_events = pd.read_csv(f'{filepath_event}/iwv_detections_2012_2024_final.csv')
    print('Detections loaded.')

    return df_events

def load_and_concatenate_mwr(filepath: str) -> xr.Dataset:
    """
    Import the MWR data from given `filepath` and load as one xr.Dataset

    Parameters:
    -----------
        - filepath: filepath to MWR data (`str`)

    Returns:
    --------
        - ds_all: 10-min resolved MWR data of IWV of LWP (`xr.Dataset`)
    """

    files_all_years = sorted(glob.glob(f"{filepath}/20[1-2][0-9]/*.nc"))
    ds_all = xr.open_mfdataset(files_all_years, combine='nested', concat_dim='time')
    ds_all.load()
    print('MWR data loaded to memory.')

    return ds_all

def _ts(x):
    """
    Sets missing time value to `pd.NaT`.

    Parameters:
    -----------
        - x: any time value

    Returns:
    --------
        - x_converted: converted x values to `pd.NaT`
    """

    x_converted = pd.to_datetime(x) if pd.notna(x) else pd.NaT

    return x_converted

def _iwv_at(ds: xr.Dataset, t, var="prw_10min", col=0):
    """
    Looks for specific IWV values for min1/min2 or peak point for MP events.

    Parameters:
    -----------
        - ds: 10-minute resolved IWV timeseries
        - t: time of specific min1/min2 or peak point
        - var: variable name of IWV data (`prw_10min`)
        - col: which statistics from ds. 0 == mean
    
    Returns:
    --------
        - da: IWV value of given time t
    """

    if pd.isna(t):
        return np.nan
    
    da = ds[var][:, col].sel(time=np.datetime64(t), method="nearest")

    return float(da.values)

def aggregate_mp_events(df_events: pd.DataFrame, ds_mwr_all_years: xr.Dataset, iwv_var: str = "prw_10min", iwv_col: int = 0) -> pd.DataFrame:
    """
    Function to merge chains of MP-M or MP-D together. Calculates new amplitudes and durations for MP-M and MP-D events.

    Parameters:
    -----------
        - df_events: `pd.DataFrame` with all identified detections from script `define_event.py`
        - ds_mwr_all_years: timeseries of 10-minute resolved IWV (and LWP) 
        - iwv_var: variable name from `ds_mwr_all_years` to select IWV
        - iwv_col: select column of IWV statistics. 0 == 10-minute means of IWV

    Returns:
    --------
        - updated df_events as csv file. Exported to specific path
        - adds also `phase` and `n_peaks` as new colums
        - `phase`: moistening or drying
        - `n_peaks`: how many detections within one event
    """

    if df_events.empty:
        out = df_events.copy()
        out["phase"] = pd.Series(dtype="object")
        out["n_peaks"] = np.nan
        return out

    df = df_events.copy()
    for c in ["time_peak","time_min1","time_min2"]:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    df = df.sort_values("time_peak").reset_index(drop=True)

    # specific order of df_events
    base_cols = ["month","time_peak","time_min1","time_min2",
                 "amp_inc","amp_dec","dur_inc_hours","dur_dec_hours",
                 "event_type","inc_extreme","dec_extreme"]

    out_records = []
    i, n = 0, len(df)

    while i < n:
        row = df.iloc[i]
        et = row["event_type"]

        # merge MP-M events together and calculate new amplitudes and durations
        if et == "MP-M":
            start_idx = i
            end_idx = i
            while end_idx + 1 < n:
                curr = df.iloc[end_idx]
                nxt  = df.iloc[end_idx + 1]
                if nxt["event_type"] != "MP-M":
                    break
                if pd.isna(curr["time_min2"]) or pd.isna(nxt["time_min1"]) or (curr["time_min2"] != nxt["time_min1"]):
                    break
                end_idx += 1

            block = df.iloc[start_idx:end_idx+1]

            t_start = _ts(block.iloc[0]["time_min1"]) # first min1 of MP chain
            t_end   = _ts(block.iloc[-1]["time_peak"]) # last peak

            iwv_start = _iwv_at(ds_mwr_all_years, t_start, iwv_var, iwv_col)
            iwv_end   = _iwv_at(ds_mwr_all_years, t_end,   iwv_var, iwv_col)
            amp_inc   = np.round(iwv_end - iwv_start, 3)
            dur_inc_h = (t_end - t_start) / np.timedelta64(1,"h") if pd.notna(t_start) and pd.notna(t_end) else np.nan

            inc_ext   = bool(block["inc_extreme"].fillna(False).any())

            rec = {
                "month": int(block.iloc[-1]["month"]),
                "time_peak": t_end,
                "time_min1": t_start,
                "time_min2": np.nan,
                "amp_inc": amp_inc,
                "amp_dec": np.nan,
                "dur_inc_hours": dur_inc_h,
                "dur_dec_hours": np.nan,
                "event_type": "MP-M",
                "inc_extreme": inc_ext,
                "dec_extreme": np.nan,
                "phase": "moistening", # new column
                "n_peaks": float(len(block)), # new column
            }
            out_records.append({**{k: rec.get(k, np.nan) for k in base_cols},
                                **{"phase": rec["phase"], "n_peaks": rec["n_peaks"]}})
            i = end_idx + 1
            continue

        # merge MP-M events together and calculate new amplitudes and durations
        if et == "MP-D":
            start_idx = i
            end_idx = i
            while end_idx + 1 < n:
                curr = df.iloc[end_idx]
                nxt  = df.iloc[end_idx + 1]
                if nxt["event_type"] != "MP-D":
                    break
                if pd.isna(curr["time_min2"]) or pd.isna(nxt["time_min1"]) or (curr["time_min2"] != nxt["time_min1"]):
                    break
                end_idx += 1

            block = df.iloc[start_idx:end_idx+1]

            t_start = _ts(block.iloc[0]["time_peak"]) # first peak of starting MP-D event
            t_end   = _ts(block.iloc[-1]["time_min2"]) # last min2 as end of MP-D event

            iwv_start = _iwv_at(ds_mwr_all_years, t_start, iwv_var, iwv_col)
            iwv_end   = _iwv_at(ds_mwr_all_years, t_end,   iwv_var, iwv_col)
            amp_dec   = np.round(iwv_start - iwv_end, 3)
            dur_dec_h = (t_end - t_start) / np.timedelta64(1,"h") if pd.notna(t_start) and pd.notna(t_end) else np.nan

            dec_ext   = bool(block["dec_extreme"].fillna(False).any())

            rec = {
                "month": int(block.iloc[0]["month"]),
                "time_peak": t_start,
                "time_min1": np.nan,
                "time_min2": t_end,
                "amp_inc": np.nan,
                "amp_dec": amp_dec,
                "dur_inc_hours": np.nan,
                "dur_dec_hours": dur_dec_h,
                "event_type": "MP-D",
                "inc_extreme": np.nan,
                "dec_extreme": dec_ext,
                "phase": "drying", # new column
                "n_peaks": float(len(block)), # new column
            }
            out_records.append({**{k: rec.get(k, np.nan) for k in base_cols},
                                **{"phase": rec["phase"], "n_peaks": rec["n_peaks"]}})
            i = end_idx + 1
            continue

        # SP remains the same, but with new information on phase and n_peaks
        rec = {k: row.get(k, np.nan) for k in base_cols}
        rec["phase"] = "moistening" if row["event_type"] == "SP-M" else "drying"
        rec["n_peaks"] = np.nan
        out_records.append(rec)
        i += 1

    out = pd.DataFrame.from_records(out_records)

    for c in ["time_peak","time_min1","time_min2"]:
        out[c] = pd.to_datetime(out[c])

    return out

def save_results_to_csv(results, filepath='/home/cbuhren/PhD/Analysis/', filename="iwv_events_2012_2024_final.csv"):
    """
    The results from the detection algorithm are declared to a pandas Dataframe
    and saved as csv file to a specific location in `{filepath}/{filename}`

    Parameters:
    -----------
        - results: `dict`, detected moistening and drying cases
        - filepath: `str`, filepath to store the csv file
        - filename: `str`, filename of detections file
    """

    df = pd.DataFrame(results)
    if not df.empty:
        df["time_peak"] = pd.to_datetime(df["time_peak"])
        df["time_min1"] = pd.to_datetime(df["time_min1"])
        df["time_min2"] = pd.to_datetime(df["time_min2"])
    df.to_csv(filepath + filename, index=False)
    print(f'File saved to {filepath}{filename}')

if __name__ == "__main__": # run script
    df_events = load_df_events_csv('/home/cbuhren/PhD/Analysis')
    ds_mwr_all_years = load_and_concatenate_mwr('/net/norte/cbuhren/data/hatpro_processed')
    df_mp_agg = aggregate_mp_events(df_events, ds_mwr_all_years, iwv_var="prw_10min", iwv_col=0)
    save_results_to_csv(df_mp_agg)