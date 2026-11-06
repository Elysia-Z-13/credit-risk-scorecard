-- ============================================================
-- 03_strategy.sql  审批阈值扫描：通过率 vs 坏账率
--
-- 这是整个项目的「决策产出」。
-- 思路：把 0.00–1.00 的阈值网格与打分结果 CROSS JOIN，
--       一次算出每个阈值下的通过率、坏账率、预期损失。
-- 依赖：data/scored.parquet（pipeline.py 生成，含 pd_hat 列）
-- ============================================================

WITH
-- 阈值网格：0.00, 0.01, ..., 1.00
thresholds AS (
    SELECT unnest(range(0, 101)) / 100.0 AS thr
),

-- 只看测试集
scored AS (
    SELECT pd_hat, bad, limit_bal
    FROM 'data/scored.parquet'
    WHERE split = 'test'
),

grid AS (
    SELECT
        t.thr                                                  AS thr,
        COUNT(*)                                               AS n_total,
        COUNT(*) FILTER (WHERE s.pd_hat <= t.thr)              AS n_approved,
        COUNT(*) FILTER (WHERE s.pd_hat >  t.thr)              AS n_rejected,
        SUM(CASE WHEN s.pd_hat <= t.thr AND s.bad = 1
                 THEN 1 ELSE 0 END)                            AS n_bad_approved,
        SUM(CASE WHEN s.pd_hat <= t.thr THEN s.limit_bal ELSE 0 END) AS amt_approved
    FROM thresholds t
    CROSS JOIN scored s
    GROUP BY t.thr
)

SELECT
    thr                                                         AS threshold,
    n_total,
    n_approved,
    n_rejected,
    ROUND(n_approved * 100.0 / n_total, 2)                      AS approval_rate_pct,
    ROUND(n_bad_approved * 100.0 / NULLIF(n_approved, 0), 2)    AS bad_rate_pct,
    ROUND(amt_approved / 1e6, 2)                                AS approved_limit_mm,

    -- 以 60% 损失率估算坏账成本（按已批额度口径）
    ROUND(
        n_bad_approved
        * (amt_approved / NULLIF(n_approved, 1))
        * 0.60 / 1e4,
        2
    )                                                           AS expected_loss_wan

FROM grid
WHERE n_approved > 0
ORDER BY thr;
