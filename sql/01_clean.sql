-- ============================================================
-- 01_clean.sql  清洗与行为特征构造
-- 输入：data/raw.parquet
-- 输出：data/clean.parquet
--
-- 数据说明（UCI Default of Credit Card Clients）
--   LIMIT_BAL    信用额度（NT$）
--   PAY_0        最近一期（9月）还款状态：-2 无消费 / -1 全额还款 / 0 循环信用 / 1..9 逾期月数
--   PAY_2..PAY_6 往前推 5 期（8月..4月）的还款状态
--   BILL_AMT1..6 各期账单金额（1 = 最近）
--   PAY_AMT1..6  各期实际还款金额（1 = 最近）
--   bad          下个月是否违约（1 = 是）
-- ============================================================

CREATE OR REPLACE VIEW raw AS SELECT * FROM 'data/raw.parquet';

-- ------------------------------------------------------------
-- 1. 基础字段 + 行为特征
--    本项目的核心：把 6 期还款状态压成一份「行为画像」
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW feat AS
SELECT
    id,
    bad,
    LIMIT_BAL                                   AS limit_bal,
    AGE                                         AS age,
    SEX                                         AS sex,
    EDUCATION                                   AS education,
    MARRIAGE                                    AS marriage,

    -- 逾期行为：6 期状态 → 次数与极值
    CASE WHEN PAY_0  > 0 THEN 1 ELSE 0 END
      + CASE WHEN PAY_2 > 0 THEN 1 ELSE 0 END
      + CASE WHEN PAY_3 > 0 THEN 1 ELSE 0 END
      + CASE WHEN PAY_4 > 0 THEN 1 ELSE 0 END
      + CASE WHEN PAY_5 > 0 THEN 1 ELSE 0 END
      + CASE WHEN PAY_6 > 0 THEN 1 ELSE 0 END            AS late_cnt,
    GREATEST(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)   AS max_late,
    PAY_0                                                AS late_cur,

    -- 额度使用：账单占额度的比例
    BILL_AMT1 / NULLIF(LIMIT_BAL, 0)                     AS util_cur,
    ((BILL_AMT1 + BILL_AMT2 + BILL_AMT3
      + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6.0)
        / NULLIF(LIMIT_BAL, 0)                           AS util_avg,
    (BILL_AMT1 - BILL_AMT6) / NULLIF(LIMIT_BAL, 0)       AS util_trend,

    -- 还款行为：还款额 / 账单额
    PAY_AMT1 / NULLIF(BILL_AMT1, 0)                      AS repay_ratio_cur,
    (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4
     + PAY_AMT5 + PAY_AMT6)
      / NULLIF(BILL_AMT1 + BILL_AMT2 + BILL_AMT3
               + BILL_AMT4 + BILL_AMT5 + BILL_AMT6, 0)   AS repay_ratio_avg,

    BILL_AMT1                                            AS bill_cur,
    (PAY_AMT1 + PAY_AMT2 + PAY_AMT3
     + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) / 6.0             AS pay_avg

FROM raw;

-- ------------------------------------------------------------
-- 2. 切分
--    本数据无日期字段，无法做严格的时间切分。
--    退而求其次：按 id 取模做确定性切分（可复现），并在报告里写清这个局限。
--    真实业务场景应当用「放款日期」做时间切分。
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW splits AS
SELECT
    *,
    CASE WHEN id % 10 < 7 THEN 'train' ELSE 'test' END AS split
FROM feat;

COPY (SELECT * FROM splits) TO 'data/clean.parquet' (FORMAT PARQUET);
