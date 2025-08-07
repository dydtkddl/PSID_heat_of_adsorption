#!/usr/bin/env python3
import warnings
from scipy.optimize import OptimizeWarning
# 1) OptimizeWarning 무시
warnings.simplefilter("ignore", OptimizeWarning)
#!/usr/bin/env python3
from scipy.optimize import OptimizeWarning
# 1) OptimizeWarning 무시

import argparse
import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

# --- 1. 인자 파싱 ---
parser = argparse.ArgumentParser(description="Parallel DSFL Fitting Script")
parser.add_argument("temperature", type=int, help="Temperature in Celsius (e.g., 10)")
parser.add_argument("num_cpus", type=int, help="Number of CPUs to use (e.g., 4)")
args = parser.parse_args()

TC       = args.temperature
NUM_CPUS = args.num_cpus

CSV_IN  = f"./data/result_O2_{TC}C_2024nahyeon_ver.csv"
CSV_OUT = f"result_O2_{TC}C_dsfl_fit_parallel.csv"

# --- 2. 압력 포인트 정의 ---
P = np.array([0.001, 0.5, 1, 5])  # bar

# --- 3. DSFL 모델 정의 ---
def dsflang(P, a1, b1, c1, a2, b2, c2):
    term1 = a1 * b1 * P / (1 + b1 * P)**c1
    term2 = a2 * b2 * P / (1 + b2 * P)**c2
    return term1 + term2

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- 4. 병렬 피팅용 함수 ---
def fit_row(row_data):
    idx, row = row_data
    # 1) 흡착량 데이터 파싱
    raw = row.iloc[1:1+4].astype(str).str.replace(r"\.+", ".", regex=True)
    q   = pd.to_numeric(raw, errors="coerce").values
    if np.isnan(q).any():
        return idx, np.nan, "DSFL", [np.nan]*6

    # 3) 초기 추정값
    qmax = np.nanmax(q)
    p0   = [qmax/2, 1.0, 1.0, qmax/2, 0.1, 1.0]

    # 4) 피팅 시도
    try:
        popt, _ = curve_fit(
            dsflang, P, q,
            p0=p0,
            bounds=(0, np.inf),
            maxfev=10000
        )
        y_pred = dsflang(P, *popt)
        err    = rmse(q, y_pred)
        return idx, err, "DSFL", popt.tolist()

    except Exception:
        return idx, np.nan, "DSFL", [np.nan]*6

# --- 5. 메인 실행부 ---
if __name__ == "__main__":
    # 5.1) CSV 로드
    df = pd.read_csv(CSV_IN)

    # 5.2) 결과 저장용 컬럼 미리 생성
    df["RMSE_DSFL"]      = np.nan
    df["model"]          = ""
    # 리스트/객체를 셀에 안전하게 저장하기 위해 dtype=object
    df["fit_parameters"] = pd.Series([None]*len(df), dtype=object)

    # 5.3) 병렬 피팅
    data = list(df.iterrows())
    results = []
    with Pool(processes=NUM_CPUS) as pool, \
         tqdm(total=len(data), desc=f"Fitting DSFL @ {TC}C", ncols=80) as pbar:
        for res in pool.imap_unordered(fit_row, data):
            results.append(res)
            pbar.update(1)

    # 5.4) 피팅 결과 DataFrame에 반영
    for idx, err, model, params in results:
        # df.at 로 단일 셀에 리스트를 객체로 저장
        df.at[idx, "RMSE_DSFL"]      = err
        df.at[idx, "model"]          = model
        df.at[idx, "fit_parameters"] = params

    # 5.5) 결과 저장
    df.to_csv(CSV_OUT, index=False)
    print(f"[✓] Done! Results saved to {CSV_OUT}")

