"""
보고서 rV0.5 그림: 시철 님 최종 모델(C2 + 공휴일 묶음 + 무료개방) 결과를 보고서 크기·색으로 다시 그린다.
입력: ../시철_결과/tables/*.csv (노트북 표), ../시철_결과/sichul_eval_rows.csv (sichul_cost.py로 만든 평가 행)
출력: ../시철_결과/figs/v05_*.png
색: 진초록 = 최종 모델, 벽돌 = 현재 방식(직전 7일 평균), 회녹색 = 나머지
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import report_charts as rc

HERE = Path(__file__).resolve().parent
R = HERE.parent / "시철_결과"
TAB = R / "tables"
OUT = R / "figs_v08"
OUT.mkdir(exist_ok=True)
rc.setup_font()
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85", "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.facecolor": "white", "figure.facecolor": "white", "axes.grid": True, "grid.color": "#ECEBE7"})
GREEN, BRICK, SAGE, SAND, INK = "#B95A45", "#7C9086", "#CFD8D2", "#DCC7A0", "#2B2B2B"   # 빨강 = 최종, 회녹 = 현재, 연회녹 = 나머지, 베이지 = 보조   # 빨강 = 최종(중요), 회색 = 현재, 연회색 = 나머지, 연빨강 = 보조   # PPT 색: 진초록 = 최종, 벽돌 = 현재, 회녹 = 나머지   # 성환 님 rV0.3 색: 주황 = 최종, 회색 = 현재, 파랑 = 나머지
LIGHT = "#DCC7A0"
FINAL, CUR = "chronos_holiday_free", "baseline_mean7"
AUD = HERE.parent / "audit"                       # guide_audit.py 결과 (직전 7일 평균 = 안내서 정의)
EV = pd.read_csv(AUD / "audit_eval_rows.csv", parse_dates=["date"])


def _cost(st, a):
    g = np.clip(st, 0, None) / 2000 - a / 2000
    return np.clip(-g, 0, None) * 6e5 + np.clip(g, 0, None) * 1.5e5


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def label_bars(ax, bars, fmt, horizontal=False, **kw):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", ha="left", fontsize=10.5, color=INK, **kw)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), va="bottom", ha="center", fontsize=10.5, color=INK, **kw)


# ---------- 그림 1: 날짜 유형별 배치 인원 · 부족 인원 · 적중률 ----------
dt = pd.read_csv(AUD / "audit_by_dtype.csv").set_index("dtype")
TYPES = ["일반 평일", "일반 주말", "일반 공휴일", "황금연휴", "설·추석 명절", "무료개방일"]
SHORTLBL = ["평일", "주말", "공휴일", "황금연휴", "명절 연휴", "무료개방일"]
dt = dt.reindex(TYPES)
f = pd.DataFrame({"need": dt["final_need"], "q80st": dt["final_q80"], "shortage_person_days": dt["final_q80_short"], "hit_rate": dt["final_hit"] / 100})
c = pd.DataFrame({"pt": dt["mean7_pt"], "shortage_person_days": dt["mean7_pt_short"], "hit_rate": 1 - dt["mean7_pt_shortrate"] / 100})
x = np.arange(len(TYPES))
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
ax = axes[0]
w = 0.27
for k, (vals, col, lab) in enumerate([(f["need"], SAND, "실제 필요"), (c["pt"], BRICK, "현재 방식"), (f["q80st"], GREEN, "최종 모델 (q80)")]):
    bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab)
ax.set_xticks(x, SHORTLBL, rotation=30, ha="right")
ax.set_title("개소당 하루 평균 인원 (명)")
for i, (n, q) in enumerate(zip(f["need"], f["q80st"])):
    ax.text(i + w, q + 0.05, f"{q / f['q80st'].iloc[0]:.1f}배", ha="center", fontsize=10, color=GREEN, fontweight="bold")
ax = axes[1]
ax.barh(x + 0.2, c["shortage_person_days"], 0.4, color=BRICK, label="현재 방식")
bars = ax.barh(x - 0.2, f["shortage_person_days"], 0.4, color=GREEN, label="최종 모델")
label_bars(ax, bars, "{:,.0f}", horizontal=True)
ax.set_yticks(x, SHORTLBL)
ax.invert_yaxis()
ax.set_title("부족 인원 (인·일)")
ax = axes[2]
ax.barh(x + 0.2, c["hit_rate"] * 100, 0.4, color=BRICK)
bars = ax.barh(x - 0.2, f["hit_rate"] * 100, 0.4, color=GREEN)
label_bars(ax, bars, "{:.0f}%", horizontal=True)
ax.axvline(80, color=INK, ls="--", lw=1)
ax.set_yticks(x, [""] * len(x))
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_title("적중률 (%, 배치 ≥ 실제)")
fig.tight_layout(rect=(0, 0.07, 1, 1))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
save(fig, "v05_fig1_date_type.png")

# ---------- 그림 2: 날짜 유형 × 10mm 이상 비 온 날 (발표자료 19장과 같은 기준, 실측 강수로 사후 구분) ----------
wx = pd.read_parquet(HERE / "weather_daily_preprocessed_2023_2025.parquet")[["date", "site", "precip_mm"]]
wx["date"] = pd.to_datetime(wx["date"])
ev2 = EV[EV.model.isin(["mean7", "final"])].merge(wx, on=["date", "site"], how="left")
ev2["dt"] = np.where(ev2["is_hol"], "공휴일", np.where(ev2["date"].dt.dayofweek >= 5, "주말", "평일"))
ev2["rain"] = np.where(ev2["precip_mm"] >= 10, "비 온 날 (10mm 이상)", "10mm 미만")
DTS = ["평일", "주말", "공휴일"]
RAINS = ["10mm 미만", "비 온 날 (10mm 이상)"]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=True)
for ax, m, title in [(axes[0], "mean7", "현재 방식 WAPE (%)"), (axes[1], "final", "최종 모델 WAPE (%)")]:
    xx = np.arange(len(DTS))
    for k, (rw, col) in enumerate(zip(RAINS, [SAND, GREEN if m == "final" else BRICK])):
        vals = []
        for dt_ in DTS:
            d = ev2[(ev2.model == m) & (ev2.dt == dt_) & (ev2.rain == rw)]
            vals.append((d.actual - d.prediction).abs().sum() / d.actual.sum() * 100 if len(d) else 0)
        bars = ax.bar(xx + (k - 0.5) * 0.38, vals, 0.38, color=col, label=rw)
        label_bars(ax, bars, "{:.0f}")
    ax.set_xticks(xx, DTS)
    ax.set_ylim(0, 95)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=10, ncol=1)
fig.tight_layout()
save(fig, "v05_fig2_rain_holiday.png")

# ---------- 그림 3: 모델별 WAPE · RelMAE ----------
ov = pd.read_csv(TAB / "01_model_comparison_overall.csv")
AO = pd.read_csv(AUD / "audit_overall.csv", index_col=0)
for mid, key in [("baseline_naive", "naive"), ("baseline_mean7", "mean7")]:      # 계절 나이브는 노트북 값(발표자료와 같게)
    ov.loc[ov.model_id == mid, "WAPE"] = AO.loc[key, "WAPE"]
ov["RelMAE"] = ov["WAPE"] / ov.loc[ov.model_id == "baseline_seasonal_naive", "WAPE"].iloc[0]
ov = ov.sort_values("WAPE")
LAB = {"baseline_naive": "나이브", "baseline_seasonal_naive": "계절 나이브", "baseline_mean7": "직전 7일 평균 (현재)", "arima": "ARIMA", "sarima": "SARIMA",
       "mlp_global": "글로벌 MLP", "chronos_zero": "C2 제로샷", "chronos_holiday": "C2 + 공휴일", "chronos_weather": "C2 + 강수", "chronos_free": "C2 + 무료개방",
       "chronos_holiday_weather": "C2 + 공휴일 + 강수", "chronos_holiday_free": "C2 + 공휴일 + 무료개방 (최종)", "chronos_weather_free": "C2 + 강수 + 무료개방",
       "chronos_holiday_weather_free": "C2 + 공휴일 + 강수 + 무료개방"}
cols = [GREEN if m == FINAL else BRICK if m == CUR else SAGE for m in ov["model_id"]]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
for ax, col, title, fmt in [(axes[0], "WAPE", "WAPE (%)", "{:.1f}"), (axes[1], "RelMAE", "RelMAE (계절 나이브 = 1)", "{:.2f}")]:
    bars = ax.barh(np.arange(len(ov)), ov[col], color=cols)
    label_bars(ax, bars, fmt, horizontal=True)
    ax.set_title(title)
    ax.set_xlim(0, ov[col].max() * 1.15)
axes[0].set_yticks(np.arange(len(ov)), [LAB[m] for m in ov["model_id"]])
axes[0].invert_yaxis()
axes[1].axvline(1, color=INK, ls="--", lw=1)
fig.tight_layout()
save(fig, "v05_fig3_models.png")

# ---------- 그림 4: FVA 폭포 (소수 기준 비용) ----------
steps = [("현재 방식\n(그대로 배치)", AO.loc["mean7", "pt_dec_total"] / 1e8), ("+ 0.8 분위수\n보정", AO.loc["mean7", "q80_dec_total"] / 1e8),
         ("Chronos-2\n제로샷", 11.4600), ("+ 공휴일 묶음", 10.3740), ("+ 무료개방\n(최종)", AO.loc["final", "q80_dec_total"] / 1e8)]
fig, ax = plt.subplots(figsize=(8.2, 3.7))
prev = None
for i, (lab, v) in enumerate(steps):
    if prev is None:
        ax.bar(i, v, 0.6, color=BRICK)
    else:
        ax.bar(i, prev - v, 0.6, bottom=v, color=SAND if i == 1 else GREEN)
        ax.text(i, prev + 0.15, f"-{prev - v:.2f}억", ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color="#8a8a85", lw=1)
    ax.text(i, v - 0.55, f"{v:.2f}억", ha="center", fontsize=11.5, color="white" if prev is None else INK)
    prev = v
ax.bar(5.2, 8.58, 0.6, color=GREEN)
ax.text(5.2, 8.58 + 0.15, "8.58억", ha="center", fontsize=12, fontweight="bold", color=GREEN)
ax.set_xticks(list(range(5)) + [5.2], [s[0] for s in steps] + ["최종 비용"])
ax.set_ylim(0, 17.6)
ax.set_ylabel("2025년 인력 비용 (억 원)")
ax.text(1, 16.9, f"보정만: -{steps[0][1] - steps[1][1]:.2f}억", ha="center", fontsize=12, color="#B95A45")
ax.text(3, 16.9, f"모델 덕분: -{steps[1][1] - steps[4][1]:.2f}억", ha="center", fontsize=12, color=GREEN)
fig.tight_layout()
save(fig, "v05_fig4_fva.png")

# ---------- 그림 5: 근로자의 날 ----------
lb = pd.read_csv(TAB / "08_labor_day_long_term_evidence.csv")
site = lb.iloc[0:23].sort_values("ratio")
year = lb.iloc[24:35]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1.2]))
ax = axes[0]
gcol = {"궁": GREEN, "종묘": BRICK, "왕릉": SAGE}
ax.barh(site["site"], site["ratio"], color=[gcol[g] for g in site["group"]])
ax.axvline(1, color=INK, ls="--", lw=1)
ax.set_title("개소별 5월 1일 ÷ 일평균 (2015~2025)")
ax.tick_params(axis="y", labelsize=8.5)
ax = axes[1]
bars = ax.bar(year["year"].astype(str), year["ratio"], color=[BRICK if r < 1 else GREEN for r in year["ratio"]])
label_bars(ax, bars, "{:.2f}")
ax.axhline(1, color=INK, ls="--", lw=1)
ax.axhline(1.61, color=GREEN, ls=":", lw=1.5)
ax.set_title("연도별 23개소 합계 배수 (점선 = 평균 1.61배)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
save(fig, "v05_fig5_labor_day.png")

# ---------- 그림 B1: 개소별 WAPE ----------
def _site(m):
    d = EV[EV.model == m]
    g_ = d.groupby("site")
    return pd.DataFrame({"WAPE": g_.apply(lambda x: (x.actual - x.prediction).abs().sum() / x.actual.sum() * 100),
                         "hit_rate": g_.apply(lambda x: (x.actual <= x.q80).mean()), "observed_visitors": g_.actual.sum(), "group": g_.group.first()})
fs = _site("final")
cs = _site("mean7").reindex(fs.index)
order = fs.assign(g=fs["group"].map({"궁": 0, "종묘": 1, "왕릉": 2})).sort_values(["g", "observed_visitors"], ascending=[True, False]).index
fs, cs = fs.loc[order], cs.loc[order]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.8), sharey=True)
yy = np.arange(len(order))
ax = axes[0]
ax.barh(yy + 0.2, cs["WAPE"], 0.4, color=BRICK, label="현재 방식")
bars = ax.barh(yy - 0.2, fs["WAPE"], 0.4, color=GREEN, label="최종 모델")
label_bars(ax, bars, "{:.0f}", horizontal=True)
ax.set_yticks(yy, [f"{s} ({g})" for s, g in zip(order, fs["group"])], fontsize=10.5)
ax.invert_yaxis()
ax.set_title("WAPE (%)")
ax.legend(frameon=False, loc="lower right")
ax = axes[1]
bars = ax.barh(yy, fs["hit_rate"] * 100, 0.6, color=[GREEN if h >= 0.8 else BRICK for h in fs["hit_rate"]])
label_bars(ax, bars, "{:.0f}%", horizontal=True)
ax.axvline(80, color=INK, ls="--", lw=1)
ax.set_xlim(0, 110)
ax.set_title("최종 모델 q80 적중률 (%, 80% 미만 = 회녹색)")
fig.tight_layout()
save(fig, "v05_figB1_site.png")

# ---------- 그림 C1: 2025 일별 합계 ----------
ev = pd.read_csv(R / "sichul_eval_rows.csv", parse_dates=["date"])
day = ev[ev["model_id"] == FINAL].groupby("date")[["actual", "prediction", "q80"]].sum() / 1000
fig, ax = plt.subplots(figsize=(9.5, 3.6))
ax.fill_between(day.index, day["prediction"], day["q80"], color=GREEN, alpha=0.18, label="점 예측 ~ q80")
ax.plot(day.index, day["actual"], color=INK, lw=1.1, label="실제")
ax.plot(day.index, day["q80"], color=GREEN, lw=1.3, ls="--", label="q80 (배치 기준)")
pk = day["actual"].idxmax()
ax.annotate(f"추석 연휴 {day.loc[pk, 'actual'] * 1000:,.0f}명", (pk, day.loc[pk, "actual"]), xytext=(-190, -45), textcoords="offset points",
            fontsize=11, color=BRICK, arrowprops=dict(arrowstyle="->", color=BRICK))
ax.set_ylabel("23개소 합계 (천 명)")
ax.legend(frameon=False, loc="upper left", ncol=1)
fig.tight_layout()
save(fig, "v05_figC1_daily.png")

# ---------- 그림 6: 근무표 확정 시점 (timing_sichul_colab.py 결과) ----------
tm = pd.read_csv(R / "timing_summary.csv")
tm = tm[(tm["basis"] == "q80") & ~tm["schedule"].str.contains("cache")]
SCH = ["매일", "1주", "2주", "4주"]
fm = tm[tm["model_id"] == FINAL].set_index("schedule").reindex(SCH)
cm = pd.read_csv(AUD / "audit_timing.csv").set_index("schedule").reindex(SCH).rename(columns={"cur_pt": "cost_dec_krw", "cur_WAPE": "WAPE"})
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
ax = axes[0]
xx = np.arange(4)
ax.bar(xx + 0.2, cm["cost_dec_krw"] / 1e8, 0.4, color=BRICK, label="현재 방식 (그대로 배치)")
bars = ax.bar(xx - 0.2, fm["cost_dec_krw"] / 1e8, 0.4, color=[GREEN if s == "2주" else LIGHT for s in SCH], label="최종 모델")
label_bars(ax, bars, "{:.2f}")
label_bars(ax, ax.patches[:4], "{:.2f}")
ax.set_xticks(xx, ["매일", "1주", "2주 (권고)", "4주"])
ax.set_ylim(0, 19)
ax.set_title("2025년 인력 비용 (억 원, 소수 기준)")
ax.legend(frameon=False, loc="upper left", fontsize=11)
ax = axes[1]
ax.plot(xx, cm["WAPE"], "o-", color=BRICK, label="현재 방식")
ax.plot(xx, fm["WAPE"], "o-", color=GREEN, label="최종 모델")
for i in range(4):
    ax.text(i, fm["WAPE"].iloc[i] - 3.2, f"{fm['WAPE'].iloc[i]:.1f}", ha="center", fontsize=10.5, color=GREEN)
    ax.text(i, cm["WAPE"].iloc[i] + 1.5, f"{cm['WAPE'].iloc[i]:.1f}", ha="center", fontsize=10.5, color=BRICK)
ax.set_xticks(xx, ["매일", "1주", "2주 (권고)", "4주"])
ax.set_ylim(15, 62)
ax.set_title("WAPE (%)")
ax.legend(frameon=False, loc="upper left", fontsize=11)
fig.tight_layout()
save(fig, "v05_fig6_timing.png")

# ---------- 그림 0: 주간 비용 (성환 님 그림 1을 최종 모델로 다시) ----------
_m = EV[EV.model == "mean7"].assign(c=lambda d: _cost(d.prediction, d.actual))
_f = EV[EV.model == "final"].assign(c=lambda d: _cost(d.q80, d.actual))
wc = pd.DataFrame({"cur": _m.groupby("week").c.sum(), "fin": _f.groupby("week").c.sum()}) / 1e6
wc["wtype"] = _m.groupby("week").wtype.first()
wc.index = pd.to_datetime(wc.index)
fig, ax = plt.subplots(figsize=(9.5, 3.3))
for wk_, t in wc["wtype"].items():
    if t != "공휴일 없는 주":
        ax.axvspan(wk_, wk_ + pd.Timedelta(days=7), color="#F3EADA" if t == "연휴 주" else "#E4EAE6", alpha=0.7, lw=0)
ax.step(wc.index, wc["cur"], where="post", color=BRICK, lw=1.8, label="현재 방식 (직전 7일 평균 그대로)")
ax.step(wc.index, wc["fin"], where="post", color=GREEN, lw=1.8, label="최종 모델 (q80)")
ax.set_ylabel("주간 인력 비용 (백만 원)")
ax.text(pd.Timestamp("2025-10-06"), wc["cur"].max() * 0.98, f" 추석 연휴 주 {wc.cur.max() / 100:.2f}억", color=BRICK, fontsize=11, va="top")
from matplotlib.patches import Patch
h, l = ax.get_legend_handles_labels()
ax.legend(h + [Patch(color="#F3EADA"), Patch(color="#E4EAE6")], l + ["연휴 주", "평일 공휴일 주"], frameon=False, loc="upper left", ncol=2, fontsize=11)
ax.set_ylim(0, wc["cur"].max() * 1.25)
fig.tight_layout()
save(fig, "v05_fig0_weekly.png")


# ---------- 그림 A1: 2026-01-01~14 일별 권고 ----------
st = pd.read_csv(TAB / "14_staffing_2026_site_daily.csv", parse_dates=["date"])
st["dec"] = np.where(st["is_regular_closed"], 0.0, st["predicted_total_q80"] / 2000)
g = st.pivot_table(index="date", columns="group", values="dec", aggfunc="sum")[["궁", "종묘", "왕릉"]]
vis = st[~st["is_regular_closed"]].groupby("date")["predicted_total_point"].sum() / 10000
intg = st.groupby("date")["final_operational_staff"].sum()
fig, ax = plt.subplots(figsize=(9.5, 4.0))
xx = np.arange(len(g))
bottom = np.zeros(len(g))
for grp, col in [("궁", GREEN), ("종묘", BRICK), ("왕릉", SAGE)]:
    ax.bar(xx, g[grp], 0.62, bottom=bottom, color=col, label=f"{grp} (q80 ÷ 2,000)")
    bottom += g[grp].to_numpy()
for i, (v, n) in enumerate(zip(bottom, intg)):
    ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, color=INK)
wd = "월화수목금토일"
ax.set_xticks(xx, [f"{d.month}/{d.day}\n{wd[d.dayofweek]}" +("*" if d.day == 1 else "") for d in g.index], fontsize=9)
ax.set_ylabel("권고 인력 (명·일)")
ax.set_ylim(0, bottom.max() * 1.3)
ax2 = ax.twinx()
ax2.plot(xx, vis.reindex(g.index), "o-", color=INK, lw=1.4, ms=4, label="예측 관람객 (만 명, 오른쪽)")
ax2.set_ylim(0, vis.max() * 1.35)
ax2.set_ylabel("예측 관람객 (만 명)")
ax2.grid(False)
ax2.spines["right"].set_visible(True)
h1_, l1 = ax.get_legend_handles_labels(); h2_, l2 = ax2.get_legend_handles_labels()
ax.legend(h1_ + h2_, l1 + l2, frameon=False, loc="upper center", ncol=4, fontsize=9)
fig.tight_layout()
save(fig, "v05_figA1_2026.png")
