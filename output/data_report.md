# Data Report

- **Source:** local file `samples/sample_sales.csv`
- **Generated:** 2026-10-20 15:26
- **Mode:** offline demo (deterministic analysis, no API key used)

## Dataset Overview

- **Rows:** 30
- **Columns:** 10

| Column | Type | Missing | Min | Max | Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| order_id | integer | 0 | 1001 | 1030 | 1015.5 |
| date | string | 0 | — | — | — |
| product | string | 0 | — | — | — |
| category | string | 0 | — | — | — |
| region | string | 0 | — | — | — |
| units | integer | 0 | 1 | 12 | 4.4 |
| unit_price | number | 0 | 25.99 | 129.99 | 63.41 |
| revenue | number | 0 | 45 | 779.94 | 262.6683 |
| customer_rating | number | 3 | 3.5 | 5 | 4.4593 |
| customer_feedback | string | 0 | — | — | — |

## Preview (first 5 rows)

| order_id | date | product | category | region | units | unit_price | revenue | customer_rating | customer_feedback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1001 | 2026-09-01 | Wireless Mouse | Accessories | North | 3 | 25.99 | 77.97 | 4.5 | Fast delivery and great build quality, very happy with this purchase |
| 1002 | 2026-09-02 | Mechanical Keyboard | Accessories | South | 5 | 79.50 | 397.50 | 5.0 | Battery life is excellent and the delivery was fast |
| 1003 | 2026-09-03 | USB-C Hub | Accessories | East | 2 | 45.00 | 90.00 | 4.0 | Good value for the price, packaging could be better |
| 1004 | 2026-09-04 | Laptop Stand | Accessories | West | 8 | 39.99 | 319.92 |  | Customer support helped me with setup, great service overall |
| 1005 | 2026-09-05 | Webcam HD 1080p | Video | North | 1 | 59.99 | 59.99 | 3.5 | The quality exceeded my expectations, delivery took two days |

## Keywords from `customer_feedback`

- quality (15)
- delivery (12)
- fast (9)
- great (9)
- battery (6)
- excellent (6)
- price (6)
- packaging (6)
- support (6)
- build (3)

## Summary

The dataset in samples/sample_sales.csv contains 30 rows and 10 columns. Column order_id ranges from 1001 to 1030 with a mean of 1015.5. Column units ranges from 1 to 12 with a mean of 4.4. Column unit_price ranges from 25.99 to 129.99 with a mean of 63.41. Column revenue ranges from 45 to 779.94 with a mean of 262.6683. Column customer_rating ranges from 3.5 to 5 with a mean of 4.4593. The most frequent terms in the customer_feedback column are quality, delivery, fast, great, battery.
