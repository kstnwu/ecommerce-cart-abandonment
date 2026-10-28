# Cart Abandonment Prediction (Working Draft)

> Working notes. To be condensed before making the repo public.
> TODO: update all results with the final run (fixed random seed).

## 1. Question and scope

**Question:** For sessions where a user adds an item to the cart, will the session end in a purchase?

**Business use:** prioritize which carts to target with abandonment reminders.

**Scope:** only sessions with at least one add-to-cart event (573,097 sessions). Purchases without a cart event are out of scope.

Why this scope: a "you left something in your cart" reminder only applies to users with items in their cart. Overall session-level purchase rate (6.8%) is a different question (e.g., homepage recommendations) and is not modeled here.

## 2. Data

- Source: [eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (Kaggle, REES46)
- File used: `2019-Oct.csv.gz` (October 2019 only), placed in `data/raw/`
- Why October: smallest file, no major promotion (November has Black Friday), and November can later serve as an out-of-time test set
- One row = one event. Event types: `view`, `cart`, `purchase`

| Metric | Value |
|---|---|
| Events | 42,448,764 |
| Users | 3,022,290 |
| Sessions | 9,244,421 |
| Time range | 2019-10-01 to 2019-10-31 (UTC) |
| Event mix | view 96.07%, cart 2.18%, purchase 1.75% |

## 3. Pipeline

```
data/raw/2019-Oct.csv.gz
  → data/processed/events_2019_10.parquet           (00_build_model_table.ipynb)
  → data/processed/model_table.parquet              (label + session + product features)
  → data/processed/feat_user_history.parquet        (user-history features)
  → notebooks/01_model.ipynb                        (models, evaluation, targeting)
```

Intermediate DuckDB tables in `00_build_model_table.ipynb`:

| Table | Grain | Purpose |
|---|---|---|
| `sessions` | one session | EDA only: cart × purchase cross-tab |
| `first_cart` | one cart session | time of first add-to-cart (reference point for label and features) |
| `cart_sessions` | one cart session | label |
| `feat_pre_cart` | one cart session | pre-cart session behavior |
| `feat_cart_product` | one cart session | attributes of the carted product |
| `model_table` | one cart session | label + features + train/test split |
| `feat_user_history` | one cart session | user behavior in earlier sessions |

## 4. Data quality

**Missing values**

| Column | Missing | % |
|---|---|---|
| `category_code` | 13,515,609 | 31.84% |
| `brand` | 6,113,008 | 14.40% |
| `user_session` | 2 | ~0% |
| all other columns | 0 | 0% |

- `category_code` is missing for a third of events → use `category_id` (no missing values) instead
- `brand` missing → kept; tested as a feature (`cart_brand_missing`), which turned out to have no predictive value

**Exact duplicate rows** (same time, type, product, user, session, price)

| Event type | Duplicates | % |
|---|---|---|
| cart | 28,073 | 3.0% |
| purchase | 76 | 0.01% |
| view | 2,071 | 0.005% |

- Duplicates concentrate in cart events (likely double clicks or multiple units added in the same second)
- Not removed: the label uses "has any purchase", which is unaffected by duplicates

## 5. Key EDA finding: many purchases bypass the cart

| has_cart | has_purchase | Sessions | % of all | % within cart group |
|---|---|---|---|---|
| 0 | 0 | 8,333,625 | 90.15% | 96.11% |
| 0 | 1 | 337,699 | 3.65% | 3.89% |
| 1 | 0 | 281,237 | 3.04% | 49.07% |
| 1 | 1 | 291,861 | 3.16% | 50.93% |

- Session-level cart conversion rate: **50.9%**
- **53.6%** of purchasing sessions have no cart event → cart-based funnel metrics miss more than half of purchases

## 6. Label

`label = 1` if the session has a purchase **after** its first add-to-cart; otherwise 0.

- 573,097 cart sessions; `label = 1`: 290,229 (50.64%) → balanced, no resampling needed
- 1,632 sessions purchased only before their first add-to-cart → `label = 0`
- 1 session with a NULL session ID is dropped

**Limitations**
- Only purchases within the same session count; a user who returns in a later session to buy is labeled 0
- Session-level label: the purchased product does not have to be the carted product

## 7. Features

**Leakage rule:** only information available at the moment of the first add-to-cart.

| Group | Features |
|---|---|
| Pre-cart session behavior | `n_views_before_cart`, `n_products_viewed_before_cart`, `secs_to_first_cart` |
| Timing | `cart_hour`, `cart_dow` |
| Carted product | `cart_price`, `cart_category_id`, `cart_brand_missing` |
| User history (earlier sessions only) | `prior_purchases`, `prior_carts`, `prior_sessions` |

**Limitation (left censoring):** data starts on Oct 1, so early-October sessions have little visible history even for returning customers.

## 8. Validation

Time-based split (no random shuffling), to mimic predicting future sessions from past data:

| Split | Period | Sessions | % label = 1 |
|---|---|---|---|
| train | Oct 1–24 | 463,222 | 50.32% |
| test | Oct 25–31 | 109,875 | 51.98% |

Baseline: predicting "purchase" for everyone gives 51.98% test accuracy.

## 9. Results

| Model | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| Logistic regression | 0.5445 | 0.5497 | 0.5376 |
| LightGBM | 0.6250 | 0.6341 | 0.5879 |
| LightGBM (no category) | 0.6164 | 0.6278 | 0.5798 |
| Logistic regression + history | 0.6955 | 0.7250 | 0.6396 |
| **LightGBM + history** | **0.7298** | **0.7508** | **0.6675** |

**Takeaways**
1. **Who the user is matters more than what they do in the session.** Three user-history features raised LightGBM ROC-AUC from 0.625 to 0.730.
2. **LightGBM beat logistic regression mainly through non-linearity, not the category feature.** Removing `cart_category_id` only lowered ROC-AUC by ~0.009.
3. **Gain importance overstates high-cardinality features.** `cart_category_id` ranked #2 by gain (36.9%) but contributed little in the ablation.
4. `secs_to_first_cart` was the most important feature by gain.

## 10. Targeting analysis

Rank test sessions by predicted abandonment probability (LightGBM + history):

| Top % targeted | Sessions | Actual abandon rate | Lift |
|---|---|---|---|
| 10% | 10,987 | 76.22% | 1.59 |
| 20% | 21,975 | 72.42% | 1.51 |
| 30% | 32,962 | 69.68% | 1.45 |
| 50% | 54,937 | 64.83% | 1.35 |
| 100% | 109,875 | 48.02% | 1.00 |

**Recommendation:** targeting the top 20% riskiest carts reaches real abandoners about 72% of the time, versus 48% for random targeting. The cutoff depends on reminder cost and budget.

**Caveat:** the model identifies who is likely to abandon, not who will respond to a reminder. Measuring reminder effectiveness requires an A/B test (next step: uplift modeling).

## 11. Next steps

- Out-of-time test on November 2019 (Black Friday)
- Add September data to reduce left censoring
- EDA on `secs_to_first_cart` vs. purchase rate
- A/B test design for cart reminders

## 12. How to reproduce

1. Download `2019-Oct.csv.gz` from the dataset page and place it in `data/raw/`
2. Create the environment: `pip install -r requirements.txt`
3. Run `notebooks/00_build_model_table.ipynb`, then `notebooks/01_model.ipynb`