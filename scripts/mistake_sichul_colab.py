# ================================================================
# 시철 님 최종 모델 설정 그대로, 한 가지만 바꿔 다시 재는 실험 (코랩, T4 GPU)
#  A. 기준       : C2 + 공휴일 묶음 + 무료개방(공식 표시)              → 노트북 표 07과 같아야 함 (WAPE 29.35%, 8.58억)
#  B. 실수 3·5   : 무료개방 표시만 '그날 유료 관람객 0명' 으로 바꿈 (미래 정보 누수)
#  C. 완벽한 예보: C2 + 공휴일 묶음 + 강수 + 무료개방에서 강수를 '그날 실측' 으로 (노트북의 평년값 버전은 WAPE 29.38%, 8.78억)
#  구간: 노트북과 같은 14일 격자 (보정 원점 13개 + 테스트 원점 27개), q80 = 예측 + 2024 하반기 오차의 0.8 분위수
#  필요한 파일 (DATA_DIR): heritage_visitors_daily_preprocessed_2023_2025_with_events.csv, sites.csv,
#                         calendar_kr_2023_2025.csv, weather_daily_preprocessed_2023_2025.parquet
# ================================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/content")
OUT_DIR = Path("/content/mistake")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_START, TEST_START, TEST_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
TIMELINE = pd.date_range(HISTORY_START, TEST_END)
TARGETS = ["domestic", "foreign_total"]
WD = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
QUANTILES = [0.1, 0.5, 0.8, 0.9]
UNIT, CU, CO = 2000, 600_000, 150_000
SEED, CHUNK, K = 42, 40, 14

v = pd.read_csv(DATA_DIR / "heritage_visitors_daily_preprocessed_2023_2025_with_events.csv", encoding="utf-8-sig", parse_dates=["date"])
v["foreign_total"] = v[["foreign_en", "foreign_ja", "foreign_zh", "foreign_etc"]].sum(axis=1, min_count=4)
sites = pd.read_csv(DATA_DIR / "sites.csv", encoding="utf-8-sig").drop_duplicates("site")
cal = pd.read_csv(DATA_DIR / "calendar_kr_2023_2025.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date").reindex(TIMELINE)
wx = pd.read_parquet(DATA_DIR / "weather_daily_preprocessed_2023_2025.parquet")
wx["date"] = pd.to_datetime(wx["date"])
SITES = sorted(sites["site"])
CLOSED_WD = sites.set_index("site")["closed_weekday"].map(WD).to_dict()
M = {t: v.pivot(index="date", columns="site", values=t).reindex(TIMELINE)[SITES] for t in ["domestic", "foreign_total", "total", "paid"]}
FILE = v.pivot(index="date", columns="site", values="is_free_admission").reindex(TIMELINE)[SITES]
P = wx.pivot(index="date", columns="site", values="precip_mm").reindex(TIMELINE)[SITES]
ACTUAL = M["total"].rename_axis("date").reset_index().melt(id_vars="date", var_name="site", value_name="actual")
hol = cal["is_holiday"].astype(int).to_numpy()
labor = TIMELINE.strftime("%m-%d").isin(["05-01"]).astype(int)


def frames(free_mode, rain):
    out = {}
    for t in TARGETS:
        out[t] = {}
        for s in SITES:
            closed = (TIMELINE.dayofweek == CLOSED_WD[s]) & (hol == 0)
            if free_mode == "file":
                flag = FILE[s].fillna(0).astype(bool).to_numpy()
            else:                                    # 실수: 그날 유료 관람객이 0명이고 문을 연 날
                flag = ((M["paid"][s] == 0) & M["total"][s].notna()).to_numpy()
            df = pd.DataFrame({"is_holiday": hol.astype(float), "is_labor_day": labor.astype(float)}, index=TIMELINE)
            if rain:                                 # 노트북 강수 묶음과 같은 4개 열, 확률 대신 그날 실측(0/1)
                p = P[s]
                heavy = (p >= 10).astype(float)
                df["rain_prob"] = (p >= 1).astype(float)
                df["rain_1_10_prob"] = ((p >= 1) & (p < 10)).astype(float)
                df["heavy_prob"] = heavy
                df["after_heavy_prob"] = ((heavy.shift(1) == 1) & (heavy == 0)).astype(float)
            df["free_open"] = (flag & ~closed).astype(float)
            df["target"] = M[t][s]
            out[t][s] = df
    return out


starts = pd.date_range(TEST_START, TEST_END, freq=f"{K}D")
cal_starts = [TEST_START - pd.Timedelta(days=K * j) for j in range(13, 0, -1)]
WIN = pd.concat([pd.DataFrame({"phase": "cal_test", "fold": [f"c{i}" for i in range(13)], "test_start": cal_starts}),
                 pd.DataFrame({"phase": "test", "fold": [f"t{i}" for i in range(len(starts))], "test_start": starts})], ignore_index=True)
WIN["origin"] = WIN["test_start"] - pd.Timedelta(days=1)
WIN["test_end"] = [min(s + pd.Timedelta(days=K - 1), TEST_END) for s in WIN["test_start"]]
WIN["steps"] = (WIN["test_end"] - WIN["test_start"]).dt.days + 1

_P = {}


def pipe():
    if "p" not in _P:
        import torch
        from chronos import Chronos2Pipeline
        torch.manual_seed(SEED); np.random.seed(SEED)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("Chronos-2 장치:", dev)
        _P["p"] = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
    return _P["p"]


def run(fr):
    covs = [c for c in next(iter(fr["domestic"].values())).columns if c != "target"]
    out = []
    for t in TARGETS:
        for steps, g in WIN.groupby("steps"):
            for i in range(0, len(g), CHUNK):
                ctx, fut = [], []
                for w in g.iloc[i:i + CHUNK].itertuples():
                    tag = f"|{w.phase}|{w.fold}"
                    dates = pd.date_range(w.test_start, w.test_end)
                    for s, f in fr[t].items():
                        past = f.loc[:w.origin, ["target"] + covs].rename_axis("timestamp").reset_index().assign(id=s + tag)
                        ctx.append(past[["id", "timestamp", "target"] + covs])
                        fut.append(f.loc[dates, covs].rename_axis("timestamp").reset_index().assign(id=s + tag)[["id", "timestamp"] + covs])
                r = pipe().predict_df(pd.concat(ctx, ignore_index=True), future_df=pd.concat(fut, ignore_index=True), prediction_length=int(steps),
                                      quantile_levels=QUANTILES, id_column="id", timestamp_column="timestamp", target="target")
                parts = r["id"].str.split("|", expand=True)
                q = np.sort(np.clip(np.column_stack([r[str(x)].to_numpy(float) for x in QUANTILES]), 0, None), axis=1)
                out.append(pd.DataFrame({"phase": parts[1].to_numpy(), "fold": parts[2].to_numpy(), "date": pd.to_datetime(r["timestamp"]).to_numpy(),
                                         "site": parts[0].to_numpy(), "target": t, "prediction": q[:, 1]}))
    return pd.concat(out, ignore_index=True)


def cq(x, lvl=0.8):
    x = np.sort(np.asarray(x, float))
    return float(x[min(int(np.ceil((len(x) + 1) * lvl)), len(x)) - 1])


def evaluate(name, pred):
    k = ["phase", "fold", "date", "site"]
    tot = (pred[pred.target == "domestic"].set_index(k)["prediction"].sort_index() + pred[pred.target == "foreign_total"].set_index(k)["prediction"].sort_index())
    tot = tot.rename("prediction").reset_index().merge(ACTUAL, on=["date", "site"], how="left")
    c = tot[tot.phase == "cal_test"].dropna(subset=["actual"]).assign(r=lambda d: d.actual - d.prediction)
    s_up = c.groupby("site")["r"].apply(cq)
    e = tot[tot.phase == "test"].dropna(subset=["actual"]).copy()
    e["q80"] = (e["prediction"] + e["site"].map(s_up)).clip(lower=0)
    g = e.q80 / UNIT - e.actual / UNIT
    cost = (np.clip(-g, 0, None) * CU + np.clip(g, 0, None) * CO).sum()
    need = (np.rint(e.actual).astype(np.int64) + UNIT - 1) // UNIT
    st = np.maximum(1, (np.rint(e.q80).astype(np.int64) + UNIT - 1) // UNIT)
    gi = st - need
    cost_int = int((np.clip(-gi, 0, None) * CU + np.clip(gi, 0, None) * CO).sum())
    return dict(실험=name, 행=len(e), WAPE=round((e.actual - e.prediction).abs().sum() / e.actual.sum() * 100, 2),
                적중률=round((e.actual <= e.q80).mean() * 100, 1), 비용_소수=round(cost), 비용_근무표=cost_int)


RES = []
for name, fm, rain in [("A 기준 (공식 무료개방)", "file", False), ("B 유료 0명 = 무료개방", "paid0", False), ("C 완벽한 예보 (강수 실측)", "file", True)]:
    t0 = time.time()
    f = frames(fm, rain)
    print(name, "| 경복궁 2025 무료개방일", int(f["domestic"]["경복궁"].loc["2025", "free_open"].sum()), flush=True)
    RES.append(evaluate(name, run(f)))
    print(RES[-1], f"{time.time() - t0:,.0f}초", flush=True)
    pd.DataFrame(RES).to_csv(OUT_DIR / "mistake_summary.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(RES).to_string(index=False))
print("검산: A가 WAPE 29.35, 근무표 비용 767,550,000이면 노트북과 같은 설정")
