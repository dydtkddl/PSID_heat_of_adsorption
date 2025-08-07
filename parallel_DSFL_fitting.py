import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

# --- 1. 설정 ---
TC = 10
CSV_IN  = f"./data/result_O2_{TC}C_2024nahyeon_ver.csv"
CSV_OUT = f"result_O2_{TC}C_dsfl_fit_parallel.csv"
NUM_CPUS = 4

P = np.array([0.001, 0.5, 1, 5])  # 압력(bar)

# --- 2. DSFL 모델 ---
def dsflang(P, a1, b1, c1, a2, b2, c2):
    term1 = a1 * b1 * P / (1 + b1 * P)**c1
    term2 = a2 * b2 * P / (1 + b2 * P)**c2
    return term1 + term2

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- 3. 병렬 피팅 함수 ---
def fit_row(row_data):
    index, row = row_data
    try:
        raw = row[1:].astype(str).str.replace(r"\.+", ".", regex=True)
        q = pd.to_numeric(raw, errors="coerce").values

        if np.isnan(q).any():
            return index, np.nan, "Dual-site Freundlich–Langmuir", [np.nan] * 6

        qmax = np.nanmax(q)
        p0 = [qmax / 2, 1.0, 1.0, qmax / 2, 0.1, 1.0]
        popt, _ = curve_fit(dsflang, P, q, p0=p0, bounds=(0, np.inf), maxfev=10000)
        y_pred = dsflang(P, *popt)
        err = rmse(q, y_pred)
        return index, err, "Dual-site Freundlich–Langmuir", popt.tolist()
    except Exception:
        return index, np.nan, "Dual-site Freundlich–Langmuir", [np.nan] * 6

# --- 4. 메인 실행부 ---
if __name__ == "__main__":
    df = pd.read_csv(CSV_IN)
    data = list(df.iterrows())

    results = []
    with Pool(processes=NUM_CPUS) as pool:
        with tqdm(total=len(data), desc="Fitting (DSFL)", ncols=80) as pbar:
            for result in pool.imap_unordered(fit_row, data):
                results.append(result)
                pbar.update(1)

    # 결과 반영
    for idx, err, model, params in results:
        df.at[idx, "RMSE_DSFL"] = err
        df.at[idx, "model"] = model
        df.at[idx, "fit_parameters"] = params

    df.to_csv(CSV_OUT, index=False)
    print(f"[✓] Done! Saved to {CSV_OUT}")
