-- ============================================================
-- 02_eda.sql  数据体检与分客群违约率
-- 依赖：data/clean.parquet
-- ============================================================

-- ------------------------------------------------------------
-- 1. 样本体检
-- ------------------------------------------------------------
SELECT
    split,
    COUNT(*)                        AS n,
    ROUND(AVG(bad) * 100, 2)        AS bad_rate_pct,
    ROUND(AVG(limit_bal), 0)        AS avg_limit,
    ROUND(AVG(age), 1)              AS avg_age,
    ROUND(AVG(late_cnt), 2)         AS avg_late_cnt
FROM 'data/clean.parquet'
GROUP BY split
ORDER BY split;

-- ------------------------------------------------------------
-- 2. 逾期次数 → 违约率（必须单调上升，否则说明特征构造有问题）
-- ------------------------------------------------------------
SELECT
    late_cnt,
    COUNT(*)                        AS n,
    ROUND(AVG(bad) * 100, 2)        AS bad_rate_pct
FROM 'data/clean.parquet'
WHERE split = 'train'
GROUP BY late_cnt
ORDER BY late_cnt;

-- ------------------------------------------------------------
-- 3. 最近一期还款状态 → 违约率
--    -1 = 全额还款，0 = 只用循环信用，1+ = 逾期月数
-- ------------------------------------------------------------
SELECT
    late_cur,
    COUNT(*)                        AS n,
    ROUND(AVG(bad) * 100, 2)        AS bad_rate_pct
FROM 'data/clean.parquet'
WHERE split = 'train'
GROUP BY late_cur
ORDER BY late_cur;

-- ------------------------------------------------------------
-- 4. 额度使用率分档 → 违约率
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN util_avg IS NULL        THEN 'NA'
        WHEN util_avg <  0.2         THEN '1_<20%'
        WHEN util_avg <  0.5         THEN '2_20-50%'
        WHEN util_avg <  0.8         THEN '3_50-80%'
        WHEN util_avg <= 1.0         THEN '4_80-100%'
        ELSE                              '5_>100%'
    END                              AS util_bucket,
    COUNT(*)                         AS n,
    ROUND(AVG(bad) * 100, 2)         AS bad_rate_pct
FROM 'data/clean.parquet'
WHERE split = 'train'
GROUP BY util_bucket
ORDER BY util_bucket;

-- ------------------------------------------------------------
-- 5. 教育程度与婚姻状况
-- ------------------------------------------------------------
SELECT
    education,
    COUNT(*)                        AS n,
    ROUND(AVG(bad) * 100, 2)        AS bad_rate_pct
FROM 'data/clean.parquet'
WHERE split = 'train'
GROUP BY education
ORDER BY education;

-- ------------------------------------------------------------
-- 6. 还款率分档（还款额 / 账单额）
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN repay_ratio_avg IS NULL     THEN 'NA'
        WHEN repay_ratio_avg <  0        THEN '1_negative'
        WHEN repay_ratio_avg <  0.1      THEN '2_<10%'
        WHEN repay_ratio_avg <  0.3      THEN '3_10-30%'
        WHEN repay_ratio_avg <  1.0      THEN '4_30-100%'
        ELSE                                  '5_>=100%'
    END                               AS repay_bucket,
    COUNT(*)                          AS n,
    ROUND(AVG(bad) * 100, 2)          AS bad_rate_pct
FROM 'data/clean.parquet'
WHERE split = 'train'
GROUP BY repay_bucket
ORDER BY repay_bucket;

-- ------------------------------------------------------------
-- 7. 缺失率体检
-- ------------------------------------------------------------
SELECT
    ROUND(AVG(CASE WHEN util_cur        IS NULL THEN 1.0 ELSE 0 END) * 100, 2) AS util_cur_null,
    ROUND(AVG(CASE WHEN repay_ratio_cur IS NULL THEN 1.0 ELSE 0 END) * 100, 2) AS repay_cur_null,
    ROUND(AVG(CASE WHEN repay_ratio_avg IS NULL THEN 1.0 ELSE 0 END) * 100, 2) AS repay_avg_null
FROM 'data/clean.parquet';
