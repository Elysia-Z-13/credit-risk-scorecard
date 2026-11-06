# -*- coding: utf-8 -*-
"""
信贷申请评分卡与审批策略分析 —— 端到端流程

数据：UCI Default of Credit Card Clients（30,000 条，23 个字段）
任务：用客户过去 6 期的还款行为，预测下个月是否违约，并给出审批阈值建议。

运行：
    .venv\\Scripts\\python.exe src\\pipeline.py

输出：
    data/raw.parquet     清洗前的原始数据
    data/clean.parquet   行为特征 + 切分
    data/scored.parquet  测试集 + 预测违约概率
    reports/tradeoff.png 通过率–坏账率权衡曲线
    reports/summary.txt  关键数字

设计要点（面试会被问）：
  1. 特征全部来自「过去 6 期已发生的行为」，不是泄露未来
  2. 做一次「去掉最近一期还款状态」的对照实验，看模型是否过度依赖单变量
  3. Logistic 与 GBM 双轨对比，用可解释性 / 上线成本判断该选谁
  4. 阈值用 SQL 的 CROSS JOIN 一次扫描，得到全局权衡曲线
"""
import os
import sys
import duckdb
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SQL 脚本里用的是相对路径（data/xxx.parquet），统一把工作目录切到项目根
os.chdir(BASE)
DATA = os.path.join(BASE, "data")
SQL = os.path.join(BASE, "sql")
REPORTS = os.path.join(BASE, "reports")

os.makedirs(REPORTS, exist_ok=True)


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def run_sql(name):
    path = os.path.join(SQL, name)
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    con = duckdb.connect()
    con.execute(sql)
    con.close()
    log(f"sql ok: {name}")


def q(sql):
    con = duckdb.connect()
    df = con.execute(sql).df()
    con.close()
    return df


# ------------------------------------------------------------------
NUM_FEATS = ["limit_bal", "age", "late_cnt", "max_late", "late_cur",
             "util_cur", "util_avg", "util_trend",
             "repay_ratio_cur", "repay_ratio_avg", "pay_avg", "bill_cur"]
CAT_FEATS = ["sex", "education", "marriage"]


def step_load():
    if not os.path.exists(os.path.join(DATA, "raw.parquet")):
        log("raw.parquet 不存在，先跑 load_raw.py")
        import load_raw
        load_raw.main()
    else:
        log("raw.parquet 已存在")


def step_clean():
    log("start clean")
    run_sql("01_clean.sql")


def step_eda():
    log("--- 样本体检 ---")
    print(q("""
        SELECT split, COUNT(*) n, ROUND(AVG(bad)*100,2) bad_rate_pct,
               ROUND(AVG(limit_bal),0) avg_limit, ROUND(AVG(late_cnt),2) avg_late_cnt
        FROM 'data/clean.parquet' GROUP BY split ORDER BY split
    """).to_string(index=False))

    log("--- 逾期次数 → 违约率 ---")
    print(q("""
        SELECT late_cnt, COUNT(*) n, ROUND(AVG(bad)*100,2) bad_rate_pct
        FROM 'data/clean.parquet' WHERE split='train'
        GROUP BY late_cnt ORDER BY late_cnt
    """).to_string(index=False))

    log("--- 最近一期还款状态 → 违约率 ---")
    print(q("""
        SELECT late_cur, COUNT(*) n, ROUND(AVG(bad)*100,2) bad_rate_pct
        FROM 'data/clean.parquet' WHERE split='train'
        GROUP BY late_cur ORDER BY late_cur
    """).to_string(index=False))

    log("--- 额度使用率分档 → 违约率 ---")
    print(q("""
        SELECT CASE
                 WHEN util_avg IS NULL THEN 'NA'
                 WHEN util_avg < 0.2   THEN '1_<20%'
                 WHEN util_avg < 0.5   THEN '2_20-50%'
                 WHEN util_avg < 0.8   THEN '3_50-80%'
                 WHEN util_avg <= 1.0  THEN '4_80-100%'
                 ELSE '5_>100%' END AS bucket,
               COUNT(*) n, ROUND(AVG(bad)*100,2) bad_rate_pct
        FROM 'data/clean.parquet' WHERE split='train'
        GROUP BY bucket ORDER BY bucket
    """).to_string(index=False))


# ------------------------------------------------------------------
# WOE / IV
# ------------------------------------------------------------------
def iv_woe_numeric(s, y, bins=8):
    try:
        b = pd.qcut(s, q=bins, duplicates="drop")
    except (ValueError, IndexError):
        return None
    t = pd.DataFrame({"bin": b, "y": y}).dropna()
    if t["bin"].nunique() < 2:
        return None
    g = t.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    return _woe_from_counts(g)


def iv_woe_cat(s, y, top=8):
    t = pd.DataFrame({"v": s.astype(str), "y": y}).dropna()
    if t.empty:
        return None
    keep = t["v"].value_counts().head(top).index
    t["v"] = np.where(t["v"].isin(keep), t["v"], "__OTHER__")
    g = t.groupby("v")["y"].agg(["count", "sum"])
    return _woe_from_counts(g)


def _woe_from_counts(g):
    g = g.copy()
    g.columns = ["n", "bad"]
    g["good"] = g["n"] - g["bad"]
    eps = 1e-6
    tb, tg = g["bad"].sum(), g["good"].sum()
    if tb == 0 or tg == 0:
        return None
    g["dist_bad"] = (g["bad"] + eps) / (tb + 2 * eps)
    g["dist_good"] = (g["good"] + eps) / (tg + 2 * eps)
    g["woe"] = np.log(g["dist_bad"] / g["dist_good"])
    iv = float(((g["dist_bad"] - g["dist_good"]) * g["woe"]).sum())
    return g["woe"], iv


def build_maps(tr, y, num_feats, cat_feats):
    iv_rows, num_maps, cat_maps = [], {}, {}
    for c in num_feats:
        r = iv_woe_numeric(tr[c], y)
        if r is None:
            continue
        w, iv = r
        num_maps[c] = w
        iv_rows.append({"feature": c, "type": "num", "iv": iv})
    for c in cat_feats:
        r = iv_woe_cat(tr[c], y)
        if r is None:
            continue
        w, iv = r
        cat_maps[c] = w
        iv_rows.append({"feature": c, "type": "cat", "iv": iv})
    tab = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)
    return num_maps, cat_maps, tab


def apply_maps(df, num_maps, cat_maps):
    out = pd.DataFrame(index=df.index)
    for c, w in num_maps.items():
        edges = np.array([b.left for b in w.index] + [w.index[-1].right])
        b = pd.cut(df[c], bins=edges, include_lowest=True)
        out[c] = b.map({iv: w.iloc[i] for i, iv in enumerate(w.index)})
    for c, w in cat_maps.items():
        out[c] = df[c].astype(str).map(w).fillna(w.get("__OTHER__", 0.0))
    return out


def ks_score(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def fit_models(tr, te, num_feats, cat_feats, tag=""):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    num_maps, cat_maps, ivtab = build_maps(tr, tr["bad"].values, num_feats, cat_feats)
    keep = ivtab.loc[ivtab["iv"] > 0.02, "feature"].tolist()
    num_maps = {k: v for k, v in num_maps.items() if k in keep}
    cat_maps = {k: v for k, v in cat_maps.items() if k in keep}

    Xtr = apply_maps(tr, num_maps, cat_maps).fillna(0.0)
    Xte = apply_maps(te, num_maps, cat_maps).fillna(0.0)
    ytr, yte = tr["bad"].values, te["bad"].values

    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

    lr = LogisticRegression(max_iter=3000, C=1.0)
    lr.fit(Xtr_s, ytr)
    p_lr_tr, p_lr_te = lr.predict_proba(Xtr_s)[:, 1], lr.predict_proba(Xte_s)[:, 1]

    gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                         max_iter=300, random_state=42)
    gbm.fit(Xtr_s, ytr)
    p_g_tr, p_g_te = gbm.predict_proba(Xtr_s)[:, 1], gbm.predict_proba(Xte_s)[:, 1]

    rows = []
    for name, ptr, pte in [(f"Logistic{tag}", p_lr_tr, p_lr_te),
                           (f"GBM{tag}", p_g_tr, p_g_te)]:
        rows.append({"model": name,
                     "auc_train": roc_auc_score(ytr, ptr),
                     "auc_test": roc_auc_score(yte, pte),
                     "ks_test": ks_score(yte, pte)})
    res = pd.DataFrame(rows)
    res["auc_gap"] = res["auc_train"] - res["auc_test"]
    return res, ivtab, keep, p_lr_te, p_g_te


def step_model():
    log("load clean data")
    df = q(f"SELECT * FROM '{os.path.join(DATA, 'clean.parquet').replace(chr(92), '/')}'")
    tr = df[df["split"] == "train"].copy()
    te = df[df["split"] == "test"].copy()
    log(f"train={len(tr):,}  test={len(te):,}  bad_rate={df['bad'].mean():.4f}")

    log("--- 全特征模型 ---")
    res, ivtab, keep, p_lr_te, p_g_te = fit_models(tr, te, NUM_FEATS, CAT_FEATS)
    print(ivtab.head(12).to_string(index=False))
    print(res.round(4).to_string(index=False))

    # 对照：去掉最近一期还款状态（late_cur），检查是否过度依赖单变量
    log("--- 对照实验：去掉 late_cur ---")
    nf2 = [c for c in NUM_FEATS if c != "late_cur"]
    res2, _, _, _, _ = fit_models(tr, te, nf2, CAT_FEATS, tag="(no late_cur)")
    print(res2.round(4).to_string(index=False))

    allres = pd.concat([res, res2], ignore_index=True)
    allres.to_csv(os.path.join(REPORTS, "model_compare.csv"), index=False)
    ivtab.to_csv(os.path.join(REPORTS, "iv_table.csv"), index=False)

    # 评分映射（PDO=20，基准 odds 1:1 对应 600 分）
    beta = 20 / np.log(2)
    alpha = 600.0
    p = np.clip(p_lr_te, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    te = te.copy()
    te["pd_hat"] = p_lr_te
    te["gbm_hat"] = p_g_te
    te["score"] = np.round(alpha - beta * logit, 0)
    out_df = te[["id", "split", "bad", "pd_hat", "gbm_hat", "score", "limit_bal"]]
    # 用 duckdb 写 parquet，避免额外依赖 pyarrow
    _con = duckdb.connect()
    _con.register("out_df", out_df)
    _con.execute(f"COPY out_df TO '{os.path.join(DATA, 'scored.parquet').replace(chr(92), '/')}' (FORMAT PARQUET)")
    _con.close()
    log("saved data/scored.parquet")

    return allres, ivtab


def step_strategy():
    log("--- 阈值扫描 ---")
    path = os.path.join(SQL, "03_strategy.sql")
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    con = duckdb.connect()
    df = con.execute(sql).df()
    con.close()

    band = df[(df["approval_rate_pct"] >= 20) & (df["approval_rate_pct"] <= 98)].copy()
    base = df.loc[df["threshold"] == 1.0].iloc[0]

    target = base["bad_rate_pct"] * 0.6
    cand = band[band["bad_rate_pct"] <= target].sort_values("approval_rate_pct", ascending=False)
    picked = cand.iloc[0] if len(cand) else band.sort_values("bad_rate_pct").iloc[0]

    print(f"\n基准（全放行）   通过率 {base['approval_rate_pct']:.1f}%  坏账率 {base['bad_rate_pct']:.2f}%")
    print(f"建议阈值 {picked['threshold']:.2f}  通过率 {picked['approval_rate_pct']:.1f}%  "
          f"坏账率 {picked['bad_rate_pct']:.2f}%  预期损失 {picked['expected_loss_wan']:.1f} 万")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(band["approval_rate_pct"], band["bad_rate_pct"], color="#1F3864", lw=2)
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Bad rate among approved (%)", color="#1F3864")
    ax.axvline(picked["approval_rate_pct"], ls="--", color="#B02A2A", lw=1)
    ax.plot(picked["approval_rate_pct"], picked["bad_rate_pct"], "o", color="#B02A2A")
    ax.annotate(f"recommended\napproval {picked['approval_rate_pct']:.1f}%\nbad {picked['bad_rate_pct']:.2f}%",
                xy=(picked["approval_rate_pct"], picked["bad_rate_pct"]),
                xytext=(picked["approval_rate_pct"] + 6, picked["bad_rate_pct"] + 1.5),
                fontsize=9, color="#B02A2A")
    ax.set_title("Approval rate vs. bad rate trade-off")
    fig.tight_layout()
    out = os.path.join(REPORTS, "tradeoff.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"saved {out}")

    df.to_csv(os.path.join(REPORTS, "threshold_sweep.csv"), index=False)
    return base, picked


def main():
    log("Step 1  load raw")
    step_load()
    log("Step 2  clean")
    step_clean()
    log("Step 3  EDA")
    step_eda()
    log("Step 4  model")
    allres, ivtab = step_model()
    log("Step 5  strategy")
    base, picked = step_strategy()

    lines = ["=== 关键数字（回填 README / 报告 / 简历）===", "", "[模型对比]"]
    for _, r in allres.iterrows():
        lines.append(f"  {r['model']:<24} AUC_train={r['auc_train']:.4f} "
                     f"AUC_test={r['auc_test']:.4f} KS={r['ks_test']:.4f} gap={r['auc_gap']:.4f}")
    lines += ["", "[Top IV]"]
    for _, r in ivtab.head(6).iterrows():
        lines.append(f"  {r['feature']:<18} IV={r['iv']:.4f}")
    lines += ["", "[策略]",
              f"  基准：通过率 {base['approval_rate_pct']:.1f}%  坏账率 {base['bad_rate_pct']:.2f}%",
              f"  建议：阈值 {picked['threshold']:.2f}  通过率 {picked['approval_rate_pct']:.1f}%  "
              f"坏账率 {picked['bad_rate_pct']:.2f}%"]
    txt = "\n".join(lines)
    with open(os.path.join(REPORTS, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
