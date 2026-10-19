# ================================================================
# 요구사항 8: 근무표 확정 시점 비교 (매일 / 1주 / 2주 / 4주) — 코랩 단독 실행용
#  시철 님 노트북(Project3_Final_Modeling_Operational_Delivery)의 정의를 그대로 옮겼다.
#   - 최종 모델: Chronos-2 제로샷 + 휴일 묶음(is_holiday, is_labor_day) + 무료개방(파일 is_free_admission, 정기 휴관일 0)
#   - 현재 방식: 직전 7일 평균 (구간 안에서는 앞서 만든 예측값을 재사용하는 재귀 규칙)
#   - 구간 보정: 같은 k일 격자의 2024-07-03~12-31 보정 구간 오차로 split conformal (개소별), q80 = 예측 + 오차의 0.8 분위수
#   - 비용: 정수 규칙(시철 님, max(1, ceil(q80/2000))) + PDF 기본 가정(소수 인원) 두 가지
#  필요한 파일 4개 (DATA_DIR에 올려 둔다):
#   heritage_visitors_daily_preprocessed_2023_2025_with_events.csv, sites.csv, calendar_kr_2023_2025.csv, pred_chronos_holiday_free.csv
# ================================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/content")                 # 파일을 올린 폴더
OUT_DIR = Path("/content/timing")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_CHRONOS = True                          # False면 2주(cache)만 계산 (검산용)

HISTORY_START, TRAIN_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
TIMELINE = pd.date_range(HISTORY_START, TEST_END)
TARGETS = ["domestic", "foreign_total"]
WEEKDAY_TO_NUMBER = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
SEASONAL_LAGS = (7, 14, 21, 28, 35, 42, 364)
QUANTILES, QUANTILE_COLUMNS = [0.1, 0.5, 0.8, 0.9], ["q10", "q50", "q80", "q90"]
COVARIATES = ["is_holiday", "is_labor_day", "free_open"]          # chronos_holiday_free 공변량 (노트북과 같은 순서)
STAFF_UNIT, SHORTAGE_COST_KRW, OVERSTAFF_COST_KRW = 2000, 600_000, 150_000
SCHEDULES = {"매일": 1, "1주": 7, "2주": 14, "4주": 28}
SEED, CHUNK = 42, 40

# ---------- 데이터 ----------
visitors = pd.read_csv(DATA_DIR / "heritage_visitors_daily_preprocessed_2023_2025_with_events.csv", encoding="utf-8-sig", parse_dates=["date"])
visitors["foreign_total"] = visitors[["foreign_en", "foreign_ja", "foreign_zh", "foreign_etc"]].sum(axis=1, min_count=4)
sites_table = pd.read_csv(DATA_DIR / "sites.csv", encoding="utf-8-sig").drop_duplicates("site")
cal_raw = pd.read_csv(DATA_DIR / "calendar_kr_2023_2025.csv", encoding="utf-8-sig", parse_dates=["date"])
SITES = sorted(sites_table["site"])
assert len(SITES) == 23 and set(visitors["site"]) == set(SITES)
CLOSED_WD = sites_table.set_index("site")["closed_weekday"].map(WEEKDAY_TO_NUMBER).to_dict()
MATRICES = {t: visitors.pivot(index="date", columns="site", values=t).reindex(TIMELINE)[SITES] for t in ["domestic", "foreign_total", "total"]}
ACTUAL = MATRICES["total"].rename_axis("date").reset_index().melt(id_vars="date", var_name="site", value_name="actual")

calendar = cal_raw.set_index("date").reindex(TIMELINE)
calendar["is_holiday"] = calendar["is_holiday"].astype(int)
calendar["is_labor_day"] = TIMELINE.strftime("%m-%d").isin(["05-01"]).astype(int)

# 코리아 그랜드 페스티벌 표시 점검 (노트북 섹션 3: 개방한 개소는 1, 정기 휴관은 0이어야 함)
fest = visitors.loc[visitors["date"].between("2025-10-29", "2025-11-09")]
bad = int(((fest["total"].notna()) & (fest["is_free_admission"] != 1)).sum() + ((fest["total"].isna()) & (fest["is_free_admission"] == 1)).sum())
print("무료개방 표시 위반 행:", bad, "(0이어야 노트북과 같은 입력)")

file_flag = visitors.pivot(index="date", columns="site", values="is_free_admission").reindex(TIMELINE)
FRAMES = {}
for target in TARGETS:
    FRAMES[target] = {}
    for s in SITES:
        closed = (TIMELINE.dayofweek == CLOSED_WD[s]) & (calendar["is_holiday"].to_numpy() == 0)
        free = (file_flag[s].fillna(0).astype(bool).to_numpy() & ~closed).astype(float)
        FRAMES[target][s] = pd.DataFrame({"is_holiday": calendar["is_holiday"].astype(float), "is_labor_day": calendar["is_labor_day"].astype(float), "free_open": free,
                                          "target": MATRICES[target][s]}, index=TIMELINE)
print("경복궁 무료개방일 수: 2023", int(FRAMES["domestic"]["경복궁"].loc["2023", "free_open"].sum()), "/ 2024", int(FRAMES["domestic"]["경복궁"].loc["2024", "free_open"].sum()),
      "/ 2025", int(FRAMES["domestic"]["경복궁"].loc["2025", "free_open"].sum()))


# ---------- 구간 ----------
def timing_windows(k):
    starts = pd.date_range(TEST_START, TEST_END, freq=f"{k}D")
    test = pd.DataFrame({"fold": [f"t{k}_{i:03d}" for i in range(1, len(starts) + 1)], "origin": starts - pd.Timedelta(days=1), "test_start": starts,
                         "test_end": [min(s + pd.Timedelta(days=k - 1), TEST_END) for s in starts], "phase": "test"})
    n_cal = int(np.ceil(182 / k))                       # 2024-07-03 무렵 ~ 2024-12-31 (2주면 노트북과 같은 13개)
    cal_starts = [TEST_START - pd.Timedelta(days=k * j) for j in range(n_cal, 0, -1)]
    cal = pd.DataFrame({"fold": [f"c{k}_{i:03d}" for i in range(1, n_cal + 1)], "origin": [s - pd.Timedelta(days=1) for s in cal_starts], "test_start": cal_starts,
                        "test_end": [s + pd.Timedelta(days=k - 1) for s in cal_starts], "phase": "cal_test"})
    return pd.concat([cal, test], ignore_index=True)


# ---------- 현재 방식 (직전 7일 평균, 노트북 baseline_next_value와 같은 규칙) ----------
def mean7_predictions(windows):
    rows = []
    for target in TARGETS:
        for w in windows.itertuples():
            dates = pd.date_range(w.test_start, w.test_end)
            for s in SITES:
                obs = MATRICES[target][s].loc[:w.origin]
                hist, last = obs.to_dict(), float(obs.dropna().iloc[-1])
                for d in dates:
                    prev = [hist.get(d - pd.Timedelta(days=j), np.nan) for j in range(1, 8)]
                    prev = [p for p in prev if np.isfinite(p)]
                    val = max(float(np.mean(prev)) if prev else last, 0.0)
                    hist[d], last = val, val
                    rows.append((w.phase, w.fold, d, s, target, val))
    return pd.DataFrame(rows, columns=["phase", "fold", "date", "site", "target", "prediction"]).assign(model_id="baseline_mean7")


# ---------- Chronos-2 ----------
_PIPE = {}


def pipeline():
    if "p" not in _PIPE:
        import torch
        from chronos import Chronos2Pipeline
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Chronos-2 장치:", device)
        _PIPE["p"] = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    return _PIPE["p"]


def chronos_predictions(windows):
    out = []
    windows = windows.assign(steps=(windows["test_end"] - windows["test_start"]).dt.days + 1)
    for target in TARGETS:
        for steps, group in windows.groupby("steps"):
            for i in range(0, len(group), CHUNK):
                ctx, fut = [], []
                for w in group.iloc[i:i + CHUNK].itertuples():
                    tag = f"|{w.phase}|{w.fold}"
                    dates = pd.date_range(w.test_start, w.test_end)
                    for s, frame in FRAMES[target].items():
                        past = frame.loc[:w.origin, ["target"] + COVARIATES].rename_axis("timestamp").reset_index().assign(id=s + tag)
                        ctx.append(past[["id", "timestamp", "target"] + COVARIATES])
                        fut.append(frame.loc[dates, COVARIATES].rename_axis("timestamp").reset_index().assign(id=s + tag)[["id", "timestamp"] + COVARIATES])
                res = pipeline().predict_df(pd.concat(ctx, ignore_index=True), future_df=pd.concat(fut, ignore_index=True), prediction_length=int(steps),
                                            quantile_levels=QUANTILES, id_column="id", timestamp_column="timestamp", target="target").sort_values(["id", "timestamp"])
                parts = res["id"].str.split("|", expand=True)
                q = np.sort(np.clip(np.column_stack([res[str(x)].to_numpy(float) for x in QUANTILES]), 0, None), axis=1)      # 0 미만 절단 + 분위수 정렬, 점 예측 = q50
                out.append(pd.DataFrame({"phase": parts[1].to_numpy(), "fold": parts[2].to_numpy(), "date": pd.to_datetime(res["timestamp"]).to_numpy(),
                                         "site": parts[0].to_numpy(), "target": target, "prediction": q[:, 1]}))
            print(f"    {target} {steps}일 구간 {len(group)}개 완료", flush=True)
    return pd.concat(out, ignore_index=True).assign(model_id="chronos_holiday_free")


def cached_chronos():
    p = pd.read_csv(DATA_DIR / "pred_chronos_holiday_free.csv", parse_dates=["date"])
    p = p.loc[p["phase"].isin(["cal_test", "test"])]
    q = np.sort(np.clip(p[QUANTILE_COLUMNS].to_numpy(float), 0, None), axis=1)
    return p.assign(prediction=q[:, 1])[["phase", "fold", "date", "site", "target", "prediction", "model_id"]]


# ---------- 평가 ----------
def conformal_value(values, level=0.8):
    v = np.sort(np.asarray(values, dtype=float))
    return float(v[min(int(np.ceil((len(v) + 1) * level)), len(v)) - 1])


def staff_int(visitors_):
    v = np.rint(np.clip(np.asarray(visitors_, float), 0, None)).astype(np.int64)
    return (v + STAFF_UNIT - 1) // STAFF_UNIT


def evaluate(name, k, pred):
    keys = ["model_id", "phase", "fold", "date", "site"]
    dom = pred.loc[pred["target"] == "domestic"].set_index(keys)["prediction"].sort_index()
    frn = pred.loc[pred["target"] == "foreign_total"].set_index(keys)["prediction"].sort_index()
    assert dom.index.equals(frn.index)
    tot = (dom + frn).rename("prediction").reset_index().merge(ACTUAL, on=["date", "site"], how="left")
    cal = tot.loc[tot["phase"] == "cal_test"].dropna(subset=["actual"]).assign(res=lambda d: d["actual"] - d["prediction"])
    par = pd.DataFrame([dict(model_id=m, site=s, s_up=conformal_value(g["res"]), q_abs=conformal_value(g["res"].abs()), n_cal=len(g)) for (m, s), g in cal.groupby(["model_id", "site"])])
    ev = tot.loc[tot["phase"] == "test"].dropna(subset=["actual"]).merge(par, on=["model_id", "site"], how="left", validate="many_to_one")
    ev["q80_equiv"] = (ev["prediction"] + ev["s_up"]).clip(lower=0)
    ev["lower_80"], ev["upper_80"] = (ev["prediction"] - ev["q_abs"]).clip(lower=0), ev["prediction"] + ev["q_abs"]
    ev["schedule"] = name
    rows = []
    for m, d in ev.groupby("model_id"):
        need_i = staff_int(d["actual"])
        for basis, col in (("점 예측", "prediction"), ("q80", "q80_equiv")):
            st_i = np.maximum(1, staff_int(d[col]))                       # 개방일 최소 1명
            sh_i, su_i = int(np.clip(need_i - st_i, 0, None).sum()), int(np.clip(st_i - need_i, 0, None).sum())
            gap = d[col].clip(lower=0).to_numpy() / STAFF_UNIT - d["actual"].to_numpy() / STAFF_UNIT
            sh_d, su_d = np.clip(-gap, 0, None).sum(), np.clip(gap, 0, None).sum()
            rows.append(dict(schedule=name, k_days=k, model_id=m, basis=basis, n_rows=len(d), min_cal_rows=int(par.loc[par["model_id"] == m, "n_cal"].min()),
                             WAPE=(d["actual"] - d["prediction"]).abs().sum() / d["actual"].sum() * 100, hit_rate=float((d["actual"] <= d["q80_equiv"]).mean()),
                             coverage=float(d["actual"].between(d["lower_80"], d["upper_80"]).mean()),
                             shortage_int=sh_i, surplus_int=su_i, cost_int_krw=sh_i * SHORTAGE_COST_KRW + su_i * OVERSTAFF_COST_KRW,
                             shortage_dec=sh_d, surplus_dec=su_d, cost_dec_krw=sh_d * SHORTAGE_COST_KRW + su_d * OVERSTAFF_COST_KRW))
    return rows, ev[["schedule", "model_id", "fold", "date", "site", "actual", "prediction", "q80_equiv", "lower_80", "upper_80"]]


ROWS, DETAIL = [], []
plan = [("2주(cache)", 14, "cache")] + ([(n, k, "run") for n, k in SCHEDULES.items()] if RUN_CHRONOS else [])
for name, k, src in plan:
    t0 = time.time()
    windows = timing_windows(k)
    ch = cached_chronos() if src == "cache" else chronos_predictions(windows)
    r, d = evaluate(name, k, pd.concat([ch, mean7_predictions(windows)], ignore_index=True))
    ROWS += r
    DETAIL.append(d)
    print(f"[{name}] 완료 {time.time() - t0:,.0f}초", flush=True)
    pd.DataFrame(ROWS).to_csv(OUT_DIR / "timing_summary.csv", index=False, encoding="utf-8-sig")          # 중간 저장

T = pd.DataFrame(ROWS)
T.to_csv(OUT_DIR / "timing_summary.csv", index=False, encoding="utf-8-sig")
pd.concat(DETAIL, ignore_index=True).to_csv(OUT_DIR / "timing_rows.csv", index=False, encoding="utf-8-sig")
c = T.loc[(T["schedule"] == "2주(cache)") & (T["basis"] == "q80")].set_index("model_id")["cost_int_krw"]
print(f"검산(2주 cache, 정수 규칙): 최종 {c['chronos_holiday_free']:,} (표 07: 767,550,000) | 현재 방식 {c['baseline_mean7']:,} (표 07: 1,210,650,000)")
print(T.loc[T["basis"] == "q80", ["schedule", "model_id", "WAPE", "hit_rate", "cost_int_krw", "cost_dec_krw"]].round(3).to_string(index=False))
