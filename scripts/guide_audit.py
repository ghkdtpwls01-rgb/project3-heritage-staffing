"""
가이드 정의대로 다시 계산 (보고서 rV0.7용)
 - 직전 7일 평균: 예측 기간 첫날 앞 7일 중 문 연 날의 평균 하나를 기간 모든 날에 씀 (가이드 「지금의 인력 배치 방식」)
 - 나이브: 원점 전 마지막 개방일 값 / 계절 나이브: 원점 전 7일 중 같은 요일 값, 휴관이면 문 연 가장 가까운 같은 요일
 - 최종 모델: 시철 님 cache 의 chronos_holiday_free (내국인 + 외국인 합)
 - 구간: 모든 모델 split conformal (2024-07-03~12-31 보정 구간 오차, 개소별), q80 = 예측 + 오차의 0.8 분위수
입력: argv[1] = cache 폴더, argv[2] = 출력 폴더
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
DL = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE   # 원자료·달력 파일이 있는 폴더
UNIT, CU, CO = 2000, 600_000, 150_000
R = {}

# ---------------- 데이터 ----------------
v = pd.read_csv(HERE / "heritage_visitors_daily_preprocessed_2023_2025_with_events.csv", encoding="utf-8-sig", parse_dates=["date"])
SITES = sorted(v["site"].unique())
GROUP = v.drop_duplicates("site").set_index("site")["group"].to_dict()
TL = pd.date_range("2023-01-01", "2025-12-31")
TOT = v.pivot(index="date", columns="site", values="total").reindex(TL)[SITES]
FREE = v.pivot(index="date", columns="site", values="is_free_admission").reindex(TL)[SITES].fillna(0)
cal = pd.concat([pd.read_csv(DL / "calendar_kr_2023_2025.csv", encoding="utf-8-sig", parse_dates=["date"]),
                 pd.read_csv(DL / "calendar_kr_2026.csv", encoding="utf-8-sig", parse_dates=["date"])]).drop_duplicates("date").set_index("date").sort_index()

ch = pd.read_csv(CACHE / "pred_chronos_holiday_free.csv", parse_dates=["date"])
ch = ch[ch["phase"].isin(["cal_test", "test"])]
q = np.sort(np.clip(ch[["q10", "q50", "q80", "q90"]].to_numpy(float), 0, None), axis=1)
ch = ch.assign(prediction=q[:, 1])
k = ["phase", "fold", "horizon", "date", "site"]
fin = (ch[ch.target == "domestic"].set_index(k)["prediction"] + ch[ch.target == "foreign_total"].set_index(k)["prediction"]).rename("prediction").reset_index()
fin["model"] = "final"
win = fin.groupby(["phase", "fold"])["date"].agg(["min", "max"]).reset_index()


# ---------------- 베이스라인 (가이드 정의) ----------------
def baselines(win):
    rows = []
    for w in win.itertuples():
        origin = w.min - pd.Timedelta(days=1)
        dates = pd.date_range(w.min, w.max)
        for s in SITES:
            hist = TOT[s].loc[:origin]
            open_ = hist.dropna()
            last7 = hist.loc[origin - pd.Timedelta(days=6):origin].dropna()
            m7 = last7.mean() if len(last7) else open_.iloc[-1]
            m7z = hist.loc[origin - pd.Timedelta(days=6):origin].fillna(0).mean()      # 실수 1: 휴관일 0을 그대로 평균
            naive = open_.iloc[-1]
            for h, d in enumerate(dates, 1):
                back = (origin - d).days % 7          # d와 같은 요일인 원점 전 7일 안의 날까지 거리
                cand = d - pd.Timedelta(days=7 * int(np.ceil((d - origin).days / 7)))
                sn = np.nan
                while cand >= TL[0]:
                    val = TOT[s].get(cand, np.nan)
                    if pd.notna(val):
                        sn = val
                        break
                    cand -= pd.Timedelta(days=7)
                rows += [(w.phase, w.fold, h, d, s, "mean7", m7), (w.phase, w.fold, h, d, s, "naive", naive),
                         (w.phase, w.fold, h, d, s, "snaive", sn), (w.phase, w.fold, h, d, s, "mean7_zero", m7z)]
    return pd.DataFrame(rows, columns=["phase", "fold", "horizon", "date", "site", "model", "prediction"])


bl = baselines(win)
P = pd.concat([fin, bl], ignore_index=True)
P["actual"] = TOT.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([P["date"], P["site"]])).to_numpy()


def cq(x, lvl=0.8):
    x = np.sort(np.asarray(x, float))
    return float(x[min(int(np.ceil((len(x) + 1) * lvl)), len(x)) - 1])


def add_q(P, lvl=0.8, col="q80"):
    cal_ = P[(P.phase == "cal_test")].dropna(subset=["actual"]).assign(r=lambda d: d.actual - d.prediction)
    par = cal_.groupby(["model", "site"])["r"].apply(lambda r: cq(r, lvl)).rename("s").reset_index()
    out = P.drop(columns=[c for c in ["s"] if c in P]).merge(par, on=["model", "site"], how="left")
    out[col] = (out["prediction"] + out["s"]).clip(lower=0)
    return out.drop(columns="s")


P = add_q(P)
cal_ = P[(P.phase == "cal_test")].dropna(subset=["actual"]).assign(r=lambda d: d.actual - d.prediction)
qa = cal_.groupby(["model", "site"])["r"].apply(lambda r: cq(np.abs(r))).rename("qa").reset_index()
P = P.merge(qa, on=["model", "site"], how="left")
E = P[(P.phase == "test")].dropna(subset=["actual"]).copy()
E["lo"], E["hi"] = (E.prediction - E.qa).clip(lower=0), E.prediction + E.qa


def cost(staff, act, integer=False, unit=UNIT, cu=CU, co=CO):
    staff, act = np.asarray(staff, float), np.asarray(act, float)
    if integer:
        need = (np.rint(act).astype(np.int64) + unit - 1) // unit
        st = np.maximum(1, (np.rint(np.clip(staff, 0, None)).astype(np.int64) + unit - 1) // unit)
    else:
        need, st = act / unit, np.clip(staff, 0, None) / unit
    g = st - need
    sh, su = np.clip(-g, 0, None).sum(), np.clip(g, 0, None).sum()
    return dict(short=sh, surp=su, short_won=sh * cu, surp_won=su * co, total=sh * cu + su * co)


def metrics(d):
    return dict(n=len(d), WAPE=(d.actual - d.prediction).abs().sum() / d.actual.sum() * 100,
                hit=(d.actual <= d.q80).mean() * 100, cover=d.actual.between(d.lo, d.hi).mean() * 100,
                width=(d.hi - d.lo).mean())


snaive_mae = None
rows = []
for m, d in E.groupby("model"):
    r = metrics(d)
    r["model"] = m
    for basis in ["prediction", "q80"]:
        for integer in [False, True]:
            c = cost(d[basis], d.actual, integer)
            tag = f"{'pt' if basis == 'prediction' else 'q80'}_{'int' if integer else 'dec'}"
            for kk, vv in c.items():
                r[f"{tag}_{kk}"] = vv
    rows.append(r)
OV = pd.DataFrame(rows).set_index("model")
sn = E[E.model == "snaive"].set_index(["fold", "date", "site"])
for m in OV.index:
    d = E[E.model == m].set_index(["fold", "date", "site"])
    OV.loc[m, "RelMAE"] = (d.actual - d.prediction).abs().sum() / (sn.actual - sn.prediction).abs().reindex(d.index).sum()
OV.to_csv(OUT / "audit_overall.csv", encoding="utf-8-sig")

# ---------------- 날짜 속성 (시철 님 정의) ----------------
off = (cal.index.dayofweek >= 5) | (cal.is_holiday == 1)
blk = pd.Series((~off).cumsum(), index=cal.index)
cal["blen"] = pd.Series(off.astype(int), index=cal.index).groupby(blk).transform("sum")
golden = pd.Series(False, index=cal.index)
for _, days in cal[off].groupby(blk[off]):
    if len(days) >= 4 and (days.is_holiday == 1).any():
        golden.loc[days.index] = True
reason = v.set_index(["date", "site"])["free_admission_reason"].fillna("")
myeong_dates = set(v.loc[v.free_admission_reason.fillna("").str.contains("설 연휴|추석 연휴"), "date"])
E["is_hol"] = cal.is_holiday.reindex(E.date).to_numpy() == 1
E["golden"] = golden.reindex(E.date).to_numpy()
E["myeong"] = E.date.isin(myeong_dates) | (E.is_hol & cal.holiday_name.fillna("").reindex(E.date).str.contains("설날|추석").to_numpy())
E["labor"] = E.date.dt.strftime("%m-%d").eq("05-01")
E["free"] = FREE.stack().reindex(pd.MultiIndex.from_arrays([E.date, E.site])).to_numpy() == 1
E["wkend"] = E.date.dt.dayofweek >= 5
E["dtype"] = np.select([E.myeong, E.golden, E.is_hol, E.labor, E.free, E.wkend],
                       ["설·추석 명절", "황금연휴", "일반 공휴일", "근로자의 날", "무료개방일", "일반 주말"], default="일반 평일")
E["group"] = E.site.map(GROUP)
cal["week"] = cal.index - pd.to_timedelta(cal.index.dayofweek, unit="D")
wk = {}
for w_, g in cal[cal.is_holiday == 1].groupby("week"):
    wk[w_] = "연휴 주" if (g.blen >= 3).any() else ("평일 공휴일 주" if (g.index.dayofweek < 5).any() else "공휴일 없는 주")
E["week"] = cal.week.reindex(E.date).to_numpy()
E["wtype"] = pd.Series(E.week).map(wk).fillna("공휴일 없는 주").to_numpy()
E.to_csv(OUT / "audit_eval_rows.csv", index=False, encoding="utf-8-sig")


def table_by(col, models=("mean7", "final")):
    out = []
    for key, d0 in E[E.model.isin(models)].groupby(col):
        r = {col: key}
        for m in models:
            d = d0[d0.model == m]
            mt = metrics(d)
            r[f"{m}_n"] = mt["n"]
            r[f"{m}_WAPE"], r[f"{m}_hit"] = mt["WAPE"], mt["hit"]
            r[f"{m}_need"] = d.actual.mean() / UNIT
            r[f"{m}_pt"] = d.prediction.mean() / UNIT
            r[f"{m}_q80"] = d.q80.mean() / UNIT
            for basis, tag in [("prediction", "pt"), ("q80", "q80")]:
                c = cost(d[basis], d.actual)
                r[f"{m}_{tag}_short"], r[f"{m}_{tag}_cost"] = c["short"], c["total"]
                r[f"{m}_{tag}_shortrate"] = (d[basis] < d.actual).mean() * 100
        out.append(r)
    return pd.DataFrame(out)


for col in ["dtype", "group", "wtype"]:
    table_by(col).to_csv(OUT / f"audit_by_{col}.csv", index=False, encoding="utf-8-sig")

# 원점별
orig = []
for f_, d0 in E.groupby("fold"):
    w = win[win.fold == f_].iloc[0]
    r = dict(fold=f_, origin=(w["min"] - pd.Timedelta(days=1)).date(), test_start=w["min"].date(), test_end=w["max"].date())
    for m in ["naive", "snaive", "mean7", "final"]:
        d = d0[d0.model == m]
        r[m] = (d.actual - d.prediction).abs().sum() / d.actual.sum() * 100
        r[m + "_cost"] = cost(d.q80 if m == "final" else d.prediction, d.actual)["total"]
    orig.append(r)
OR = pd.DataFrame(orig)
OR.to_csv(OUT / "audit_by_origin.csv", index=False, encoding="utf-8-sig")

# 무료개방·공휴일 뺀 날
ex = E[~(E.free | E.is_hol)]
R["excl_free_hol"] = {m: metrics(d)["WAPE"] for m, d in ex.groupby("model")}
R["excl_n"] = int((ex.model == "final").sum())
# 실수 1: 휴관일 0
R["mistake_zero"] = {m: metrics(E[E.model == m])["WAPE"] for m in ["mean7", "mean7_zero"]}
R["mistake_zero_cost"] = {m: cost(E[E.model == m].prediction, E[E.model == m].actual)["total"] for m in ["mean7", "mean7_zero"]}
# 실수 4: 합계로 평가
f = E[E.model == "final"]
agg = f.groupby(["fold", "date"])[["actual", "prediction"]].sum()
R["mistake_sum"] = {"site_level": metrics(f)["WAPE"], "summed": (agg.actual - agg.prediction).abs().sum() / agg.actual.sum() * 100}
# 특별 무료개방 제외, Chronos 공개 이후
fest = E.date.between("2025-10-29", "2025-11-09")
R["excl_festival"] = {"cur": cost(E[(E.model == "mean7") & ~fest].prediction, E[(E.model == "mean7") & ~fest].actual)["total"],
                      "fin": cost(E[(E.model == "final") & ~fest].q80, E[(E.model == "final") & ~fest].actual)["total"]}
late = E.fold.str[-2:].astype(int) >= 22
R["post_release"] = {m: metrics(E[(E.model == m) & late])["WAPE"] for m in ["mean7", "final"]}
R["wins"] = {"final<mean7": int((OR.final < OR.mean7).sum()), "final<snaive": int((OR.final < OR.snaive).sum())}

# 황금연휴 보정 규칙 (궁·종묘 +2명 = +4,000명) 전후, 전체 비용
f = E[E.model == "final"].copy()
mask = f.golden & f.group.isin(["궁", "종묘"])
f["q80_rule"] = f.q80 + np.where(mask, 2 * UNIT, 0)
R["rule"] = {"n_rows": int(mask.sum()),
             "before_dec": cost(f.q80, f.actual), "after_dec": cost(f.q80_rule, f.actual),
             "before_int": cost(f.q80, f.actual, True)["total"], "after_int": cost(f.q80_rule, f.actual, True)["total"],
             "hit_before": float((f[mask].actual <= f[mask].q80).mean() * 100), "hit_after": float((f[mask].actual <= f[mask].q80_rule).mean() * 100),
             "golden_cost_before": cost(f[mask].q80, f[mask].actual)["total"], "golden_cost_after": cost(f[mask].q80_rule, f[mask].actual)["total"]}

# 가정 바꾸기 (분위수 수준도 임계비율에 맞춤)
sc = []
for name, unit, cu, co in [("기본", 2000, CU, CO), ("1명=1,500명", 1500, CU, CO), ("1명=2,500명", 2500, CU, CO), ("부족 90만", 2000, 900_000, CO), ("과잉 30만", 2000, CU, 300_000)]:
    lvl = cu / (cu + co)
    PP = add_q(P[P.model.isin(["mean7", "final"])], lvl, "qx")
    TT = PP[(PP.phase == "test")].dropna(subset=["actual"])
    c_cur = TT[TT.model == "mean7"]
    c_fin = TT[TT.model == "final"]
    rr = dict(scenario=name, level=lvl,
              cur_pt=cost(c_cur.prediction, c_cur.actual, unit=unit, cu=cu, co=co)["total"],
              cur_q=cost(c_cur.qx, c_cur.actual, unit=unit, cu=cu, co=co)["total"],
              fin=cost(c_fin.qx, c_fin.actual, unit=unit, cu=cu, co=co)["total"],
              fin_q80=cost(c_fin.q80, c_fin.actual, unit=unit, cu=cu, co=co)["total"],
              staff_2026=None)
    sc.append(rr)
SC = pd.DataFrame(sc)
SC["saving"] = SC.cur_pt - SC.fin
SC.to_csv(OUT / "audit_scenarios.csv", index=False, encoding="utf-8-sig")

# 2026 권고 인원이 가정에 따라 얼마나 바뀌는지 (q80 고정 근사: 인원 = q / 단위)
st = pd.read_csv(HERE.parent / "시철_결과" / "tables" / "14_staffing_2026_site_daily.csv")
o = st[~st.is_regular_closed]
R["staff2026"] = {"2000": float((o.predicted_total_q80 / 2000).sum()), "1500": float((o.predicted_total_q80 / 1500).sum()),
                  "2500": float((o.predicted_total_q80 / 2500).sum())}

# ---------------- 확정 시점: 현재 방식(가이드 정의) ----------------
TS = pd.read_csv(HERE.parent / "results_timing_sichul" / "timing_summary.csv")
tim = []
for name, kd in {"매일": 1, "1주": 7, "2주": 14, "4주": 28}.items():
    starts = pd.date_range("2025-01-01", "2025-12-31", freq=f"{kd}D")
    tw = pd.DataFrame({"phase": "test", "fold": [f"t{i}" for i in range(len(starts))], "min": starts,
                       "max": [min(s + pd.Timedelta(days=kd - 1), pd.Timestamp("2025-12-31")) for s in starts]})
    n_cal = int(np.ceil(182 / kd))
    cs = [pd.Timestamp("2025-01-01") - pd.Timedelta(days=kd * j) for j in range(n_cal, 0, -1)]
    cw = pd.DataFrame({"phase": "cal_test", "fold": [f"c{i}" for i in range(n_cal)], "min": cs, "max": [s + pd.Timedelta(days=kd - 1) for s in cs]})
    B = baselines(pd.concat([cw, tw]))
    B = B[B.model == "mean7"].copy()
    B["actual"] = TOT.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([B["date"], B["site"]])).to_numpy()
    B = add_q(B)
    T_ = B[B.phase == "test"].dropna(subset=["actual"])
    fr = TS[(TS.schedule == name) & (TS.model_id == "chronos_holiday_free") & (TS.basis == "q80")].iloc[0]
    tim.append(dict(schedule=name, n=len(T_), cur_WAPE=(T_.actual - T_.prediction).abs().sum() / T_.actual.sum() * 100,
                    cur_pt=cost(T_.prediction, T_.actual)["total"], cur_q=cost(T_.q80, T_.actual)["total"], cur_q_int=cost(T_.q80, T_.actual, True)["total"],
                    cur_pt_int=cost(T_.prediction, T_.actual, True)["total"],
                    fin_WAPE=fr.WAPE, fin_hit=fr.hit_rate * 100, fin_dec=fr.cost_dec_krw, fin_int=fr.cost_int_krw))
pd.DataFrame(tim).to_csv(OUT / "audit_timing.csv", index=False, encoding="utf-8-sig")

# ---------------- 데이터 주의 사항 (원자료) ----------------
raw = pd.read_parquet(DL / "heritage_visitors_daily.parquet")
sites = pd.read_csv(DL / "sites.csv")
cw_ = sites.set_index("site")["closed_weekday"].map({"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6})
raw["wd"] = raw.date.dt.dayofweek
raw["closed_day"] = raw.wd == raw.site.map(cw_)
z = raw.total == 0
dc = {"rows": len(raw), "zero_rows": int(z.sum()),
      "zero_on_closed_weekday": int((z & raw.closed_day).sum()), "zero_not_closed_weekday": int((z & ~raw.closed_day).sum()),
      "open_on_closed_weekday": int((~z & raw.closed_day).sum()),
      "paid_free_mismatch": int((raw.paid + raw.free != raw.total).sum()),
      "dom_for_mismatch": int((raw.domestic.astype(float) + raw[["foreign_en", "foreign_ja", "foreign_zh", "foreign_etc"]].sum(axis=1) != raw.total).sum()),
      "negative": int((raw[["paid", "free", "foreign_en", "foreign_ja", "foreign_zh", "foreign_etc", "total"]] < 0).sum().sum()),
      "null": int(raw.isna().sum().sum()), "null_domestic": int(raw.domestic.isna().sum())}
gaps = {}
for s_, g in raw.groupby("site"):
    full = pd.date_range(g.date.min(), g.date.max())
    gaps[s_] = int(len(full) - g.date.nunique())
dc["missing_dates_total"] = sum(gaps.values())
dc["missing_by_site"] = {k_: v_ for k_, v_ in gaps.items() if v_}
dc["start_year"] = raw.groupby("site").date.min().dt.year.value_counts().to_dict()
open_closed = raw[~z & raw.closed_day]
dc["open_on_closed_weekday_holiday_share"] = float(cal.is_holiday.reindex(open_closed.date).fillna(0).mean()) if len(open_closed) else None
pz = raw[(raw.paid == 0) & (raw.total > 0)]
fsh = lambda d: (d[["foreign_en", "foreign_ja", "foreign_zh", "foreign_etc"]].sum() / d[["foreign_en", "foreign_ja", "foreign_zh", "foreign_etc"]].sum().sum()).round(3).to_dict()
dc["lang_share_paid0"] = fsh(pz)
dc["lang_share_all"] = fsh(raw[raw.total > 0])
yr = raw[raw.group == "궁"].groupby(raw.date.dt.year).total.sum()
dc["palace_2019_2025"] = {"2019": int(yr.get(2019, 0)), "2020": int(yr.get(2020, 0)), "2023": int(yr.get(2023, 0)), "2025": int(yr.get(2025, 0))}
R["data_checks"] = dc
json.dump(R, open(OUT / "audit_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("done")
