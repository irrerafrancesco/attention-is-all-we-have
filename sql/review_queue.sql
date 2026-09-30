-- 1. Top 100 customers by predicted risk
SELECT
    customer_id,
    review_rank,
    predicted_risk,
    actual_outcome
FROM review_queue
ORDER BY predicted_risk DESC
LIMIT 100;


-- 2. Overall queue statistics
SELECT
    COUNT(*) AS customers_in_queue,
    AVG(predicted_risk) AS average_predicted_risk,
    SUM(actual_outcome) AS actual_problem_cases
FROM review_queue;


-- 3. Customers with predicted risk >= 50%
SELECT
    COUNT(*) AS high_risk_customers
FROM review_queue
WHERE predicted_risk >= 0.50;


-- 4. Risk bands
SELECT
    CASE
        WHEN predicted_risk >= 0.70 THEN 'Very High'
        WHEN predicted_risk >= 0.50 THEN 'High'
        WHEN predicted_risk >= 0.30 THEN 'Medium'
        ELSE 'Lower'
    END AS risk_band,

    COUNT(*) AS customers,
    ROUND(AVG(predicted_risk), 4) AS average_predicted_risk,
    SUM(actual_outcome) AS actual_problem_cases,

    ROUND(
        100.0 * SUM(actual_outcome) / COUNT(*),
        2
    ) AS observed_problem_rate_pct

FROM review_queue

GROUP BY risk_band

ORDER BY average_predicted_risk DESC;