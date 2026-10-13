# Recipe Recommender: Collaborative Filtering on Food.com Recipes & Interactions

A machine learning recommendation engine evaluating neighborhood-based and matrix factorization collaborative filtering algorithms on the **Food.com (formerly GeniusKitchen)** dataset using Python and the **Scikit-Surprise** framework.

---

## Repository Structure

As designed, the project maintains a clean, minimalist structure:
```text
Food.com Recipe Recommendation System/
├── data/                                 # All raw and preprocessed Food.com dataset files
│   ├── RAW_interactions.csv              # 1,132,367 user reviews and rating interactions
│   ├── RAW_recipes.csv                   # 231,637 recipe metadata, tags, nutrition, steps
│   ├── PP_users.csv                      # Preprocessed user interaction vectors
│   ├── PP_recipes.csv                    # Tokenized recipe text metadata
│   ├── ingr_map.pkl                      # Ingredient normalization dictionary (11,659 -> 8,023)
│   ├── interactions_train.csv            # Training partition
│   ├── interactions_validation.csv       # Validation partition
│   └── interactions_test.csv             # Test partition
├── Recipe_Recommender.ipynb              # Complete end-to-end Jupyter Notebook
└── README.md                             # Comprehensive project documentation & report
```

---

## Dataset References & Background

- **Kaggle Dataset**: [Food.com Recipes and User Interactions](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) (180K+ recipes, 700K+ reviews spanning 18 years).
- **Academic Paper**: Bodhisattwa Prasad Majumder\*, Shuyang Li\*, Jianmo Ni, Julian McAuley. *Generating Personalized Recipes from Historical User Preferences*, EMNLP, 2019 ([ACL Anthology](https://aclanthology.org/D19-1613/)).
- **Project Summary Video**: [Food.com Recommendation System Overview](https://www.youtube.com/watch?v=gAa4Jx_Zu7I)

---

## I. Project Goals and Tasks

In the space of online food recipes, there are seemingly endless possibilities of new recipes to try. This project seeks to provide accurate recipe rating predictions and recipe recommendations using various collaborative filtering methods.

### Primary Objectives:
1. **Exploratory Data Analysis (EDA)**: Understand user review behavior, rating score distributions, recipe preparation metrics, ingredient normalization patterns, and meal types.
2. **Data Cleaning & Sparsity Reduction**: Systematically clean ambiguous ratings and filter extreme matrix sparsity to enable scalable model training.
3. **Collaborative Filtering Benchmark**:
   - **Neighborhood Models**: Item-based and user-based neighborhood algorithms (`KNNBasic`, `KNNWithMeans`, `KNNWithZScore`, `KNNBaseline`) and `SlopeOne`.
   - **Matrix Factorization Models**: `NMF` (Non-negative Matrix Factorization), `SVD` (Funk SVD), and `SVD++` (accounting for implicit feedback).
4. **Model Evaluation**: Benchmark models using 5-fold cross-validation on Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and Mean Absolute Error (MAE).
5. **Hyperparameter Optimization**: Perform Grid Search (`GridSearchCV`) to tune the top-performing neighborhood model and the top practical matrix factorization model.
6. **Recommendation Generation & Qualitative Audit**: Generate top-N personalized recommendations for active users and evaluate recommendation quality against historical meal-type preferences.

---

## II. Exploratory Data Analysis (EDA) Summary

| Dimension | Observation & Summary |
| :--- | :--- |
| **User Activity** | Median ratings per user is **6**. **73.38%** of users rated only **1 recipe**; only **2.31%** rated >25 recipes. |
| **Recipe Frequency** | **39.70%** of recipes received only 1 rating; only **2.39%** received >25 ratings. |
| **Rating Distribution** | Heavily skewed positive: **72.09%** are 5-star, **16.55%** are 4-star, **3.61%** are 3-star, **1.25%** are 2-star, **1.13%** are 1-star, and **5.37%** are 0. |
| **Zero Ratings** | Ratings of `0` were found to represent questions or observations rather than low ratings (e.g., *"i made it and it was amazing"*, *"Just an observation, so I will not rate"*), requiring removal. |
| **Cooking Duration** | Most recipes require **20 to 65 minutes** to prepare (median: 40 min). |
| **Recipe Complexity** | Most recipes consist of **6 to 12 ingredients** and **6 to 11 steps**. |
| **Ingredient Vocabulary** | 11,659 raw ingredient names map down to **8,023 normalized ingredients** in `ingr_map.pkl`. 75% of ingredients appear in $\le 50$ recipes. Top staples: salt, butter, egg, onion, sugar, olive oil, garlic clove, water, milk, flour. |
| **Meal Categorization** | Tags mapped into meal types: Main Dish (**31%**), Desserts (**19%**), Side Dishes (**10%**), Lunch (**6%**), Appetizers (**5%**), Other (**5%**), Beverages (**4%**), Salads (**3%**), Soups/Stews (**3%**), etc. |

---

## III. Description of Approach

### 1. Data Cleaning & Sparsity Reduction
- **Step 1: Removing 0 Ratings**: Dropped 60,847 ambiguous 0 ratings, reducing the interaction set from 1,132,367 to 1,071,520 reviews.
- **Step 2: Popular Recipes Filter**: Filtered to recipes with $> 25$ ratings (reduced to 341,547 ratings).
- **Step 3: Active Users Filter**: Filtered to users with $> 25$ ratings (reduced to 133,813 ratings across 2,089 users and 5,102 recipes).
- **Step 4: Re-filtering Sparse Recipes**: Filtered to recipes with $> 5$ ratings in the active subset, resulting in a final high-confidence dataset of **132,928 ratings** covering **2,089 users** and **4,841 recipes**.

### 2. Modeling Framework
Models were trained using the **Scikit-Surprise** library (`Reader(rating_scale=(1, 5))`) with **5-fold cross-validation** evaluating:
- **Mean Absolute Error (MAE)** *(Primary Metric)*: Robust against non-outlier integer ratings.
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.
- **Fit Time & Test Time**: Critical for deployment feasibility.

---

## IV. Results: Collaborative Filtering

### 1. Neighborhood Models Benchmark (5-Fold CV)

| Model | Similarity / Based | RMSE | MSE | MAE | Fit Time (s) | Test Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KNNBasic** | **Item-Based** | **0.5723** | **0.3276** | **0.3463** | 2.01 | 4.20 |
| **KNNWithZScore** | User-Based | 0.5649 | 0.3191 | 0.3489 | 0.81 | 2.75 |
| **KNNBaseline** | Item-Based | 0.5603 | 0.3139 | 0.3502 | 2.02 | 4.14 |
| **KNNBaseline** | User-Based | 0.5587 | 0.3121 | 0.3532 | 0.96 | 2.77 |
| **KNNWithMeans** | User-Based | 0.5579 | 0.3113 | 0.3532 | 0.61 | 2.64 |
| **KNNWithZScore** | Item-Based | 0.5754 | 0.3311 | 0.3574 | 2.27 | 5.16 |
| **SlopeOne** | Item Baseline | 0.5688 | 0.3235 | 0.3604 | 1.74 | 2.96 |
| **KNNWithMeans** | Item-Based | 0.5675 | 0.3221 | 0.3607 | 1.91 | 5.02 |
| **KNNBasic** | User-Based | 0.6015 | 0.3618 | 0.3696 | 0.58 | 2.44 |

> **Takeaway**: Item-based `KNNBasic` achieved the best neighborhood MAE (**0.3463**). Item rating distributions are more consistent than user rating habits.

---

### 2. Matrix Factorization Benchmark (5-Fold CV)

| Model | Biased Configuration | RMSE | MSE | MAE | Fit Time (s) | Test Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SVD++** | Biased (Implicit Feedback) | **0.5482** | **0.3005** | **0.3481** | **218.30** | 4.33 |
| **NMF** | `biased = True` | 0.5490 | 0.3014 | 0.3582 | 12.81 | 0.38 |
| **SVD** | `biased = True` | **0.5512** | **0.3039** | **0.3526** | **8.28** | **0.27** |
| **NMF** | `biased = False` | 0.6395 | 0.4090 | 0.5066 | 10.04 | 0.29 |
| **SVD** | `biased = False` | 0.6566 | 0.4312 | 0.5207 | 8.44 | 0.25 |

> **Takeaway**: 
> - **Bias adjustment is paramount**: Incorporating baseline biases drops MAE from ~0.51 down to ~0.35.
> - While `SVD++` achieved the lowest MAE (**0.3481**), its fit time was **218 seconds**. Regular `SVD` achieved a comparable **0.3526 MAE** in only **8.28 seconds** (>26× faster), making `SVD` the superior choice for production scalability.

---

### 3. Hyperparameter Optimization via GridSearchCV

#### A. Tuning KNNBasic (Item-Based)
- **Parameters Tested**: Similarity measures (`MSD`, `Cosine`, `Pearson`), $k \in [5, 10, 13, 20, 30]$, $\text{min\_k} \in [3, 5]$.
- **Optimal Parameters**: $\text{Metric} = \text{MSD}$, $k = 13$, $\text{min\_k} = 5$.
- **Tuned 5-Fold MAE**: **0.3548**.

#### B. Tuning SVD Matrix Factorization
- **Parameters Tested**:
  - `n_factors`: $50, 100$
  - `n_epochs`: $10, 20$
  - `biased`: $\text{True}, \text{False}$
  - `lr_all`: $0.0005, 0.005, 0.05$
  - `reg_all`: $0.002, 0.02, 0.2$
- **Optimal Parameters**:
  - `n_factors` = 50
  - `n_epochs` = 20
  - `biased` = True
  - `lr_all` = 0.005
  - `reg_all` = 0.02
- **Tuned 5-Fold MAE**: **0.3491**.

---

### 4. Recommendation Generation & Qualitative Audit

Using the tuned SVD model, top-10 recommendations were generated for unrated recipes for representative active users:

```text
Representative Top-10 Recommendations:
1. "to die for crock pot roast"                       (Predicted: 4.88) | Meal: main-dish
2. "crock pot chicken with black beans & cream cheese"(Predicted: 4.85) | Meal: main-dish
3. "creamy cajun chicken pasta"                       (Predicted: 4.81) | Meal: main-dish
4. "jo mama s world famous spaghetti"                 (Predicted: 4.80) | Meal: main-dish
5. "kitten s soft snickerdoodles"                     (Predicted: 4.78) | Meal: desserts
6. "best banana bread"                                (Predicted: 4.77) | Meal: desserts
7. "awesome broccoli cheese soup"                     (Predicted: 4.76) | Meal: soups-stews
8. "brown sugar meatloaf"                             (Predicted: 4.76) | Meal: main-dish
9. "beth s melt in your mouth sugar cookies"          (Predicted: 4.75) | Meal: desserts
10. "fabulous french onion soup"                      (Predicted: 4.74) | Meal: soups-stews
```

---

## V. Key Insights & Limitations

1. **Numerical Accuracy vs. Recommendation Utility**:
   - The numerical accuracy is strong: an MAE of $\approx 0.35$ on integer ratings means rounding predicted ratings produces the exact true rating for most interactions.
2. **Popularity Bias**:
   - Matrix factorization generates nearly identical top-10 recommendation lists across different users, dominated by globally high-volume, 5-star recipes.
3. **Meal-Type Mismatch**:
   - Comparing users' historical profiles against top recommendations reveals frequent meal-type discrepancies (e.g., suggesting desserts or beef roasts to users who predominantly review light vegetarian lunches or side dishes).
4. **Rating Compression**:
   - Because 88.6% of ratings in Food.com are 4 or 5 stars, predicted ratings cluster tightly between 4.70 and 4.90, making relative ranking sensitive to tiny numerical variations.

---

## VI. Conclusion & Future Directions

Pure collaborative filtering via ratings alone is insufficient for a personalized culinary recommender. Future iterations should incorporate:
1. **Hybrid Filtering**: Blend matrix factorization latent factors with content-based features (ingredient TF-IDF, preparation times, and calorie levels from `PP_recipes.csv`).
2. **Meal-Type & Dietary Guardrails**: Filter or re-rank candidate recipes based on the user's historical meal-type distribution, dietary restrictions, and allergies.
3. **Implicit Feedback Signals**: Supplement explicit ratings with recipe saves, bookmarks, and search interactions to counteract rating inflation.

---

## How to Run

### Requirements
- Python 3.8+
- Required packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-surprise
  ```

### Running the Notebook
1. Open the repository root:
   ```bash
   jupyter notebook Recipe_Recommender.ipynb
   ```
2. The notebook automatically detects dataset files in the `data/` directory and executes sequentially through ingestion, EDA, filtering, modeling, tuning, and recommendation generation.
