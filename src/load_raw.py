# -*- coding: utf-8 -*-
"""
把 UCI 原始数据读成干净的 parquet，交给后面的 SQL 处理。

数据：UCI Default of Credit Card Clients
      台湾某银行 2005 年信用卡客户，30,000 条，23 个字段
      标签：default payment next month（下个月是否违约）

原始文件有两种可能：
  - _uci/default of credit card clients.xls   （Excel，表头在第 2 行）
  - credit-card-clients.csv                   （CSV，表头在中英文两行）
本脚本自动识别。
"""
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

XLS_CANDIDATES = [
    os.path.join(DATA, "_uci", "default of credit card clients.xls"),
    os.path.join(DATA, "_uci", "default of credit card clients.csv"),
    os.path.join(DATA, "credit-card-clients.csv"),
    os.path.join(DATA, "loan.csv"),
]

RENAME = {
    "default payment next month": "bad",
    "default.payment.next.month": "bad",
    "ID": "id",
}


def find_raw():
    for p in XLS_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def read_any(path):
    """读 xls / csv。原始文件有两种表头形态，自动判断：
       - 单行英文表头（GitHub 镜像的 csv）→ header=0
       - 中英文双表头（UCI 官方 csv）或 xls 的第 2 行 → header=1
    """
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, header=1)
    else:
        peek = pd.read_csv(path, nrows=1)
        cols = [str(c).strip().upper() for c in peek.columns]
        single = any(c in ("ID", "LIMIT_BAL") for c in cols)
        df = pd.read_csv(path, header=0 if single else 1)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main():
    raw = find_raw()
    if raw is None:
        print("[load_raw] 找不到原始数据，请检查 data/ 目录：")
        print("           期望文件：data/_uci/default of credit card clients.xls")
        sys.exit(1)

    print(f"[load_raw] 读取 {raw}")
    df = read_any(raw)
    print(f"[load_raw] 原始形状 {df.shape}")
    print(f"[load_raw] 前 3 列：{list(df.columns[:3])}")

    # 统一列名
    df = df.rename(columns=RENAME)
    if "bad" not in df.columns:
        # 最后一列就是标签
        df = df.rename(columns={df.columns[-1]: "bad"})
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # 类型清洗
    for c in df.columns:
        if c != "id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df[df["bad"].isin([0, 1])].copy()
    print(f"[load_raw] 有效标签 {len(df):,} / {before:,}，坏账率 {df['bad'].mean():.4f}")

    out = os.path.join(DATA, "raw.parquet")
    # 用 duckdb 写 parquet，避免额外依赖 pyarrow
    import duckdb
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY df TO '{out}' (FORMAT PARQUET)")
    con.close()
    print(f"[load_raw] saved {out}")


if __name__ == "__main__":
    main()
