# Recipe Recommender System

### Collaborative Filtering, Clustering, and Content Based Recommendations on Food.com

**Author: Vidhya Walke**

## Project Overview

In the space of online food recipes, there are seemingly endless possibilities of new recipes to try. This project develops an end to end recipe recommendation engine that predicts recipe ratings and suggests personalized recipes to users.

The system explores three distinct recommendation paradigms:
1. User and Item Based Collaborative Filtering using Scikit Surprise (Neighborhood and Matrix Factorization models)
2. Cluster Based Collaborative Filtering using CoClustering, KMeans, and NLTK Cosine Clustering
3. Content Based Recommendation using TF IDF vectorization on recipe descriptions evaluated by Precision and Recall

## Dataset Description

Kaggle Dataset Link: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

The analysis uses the Food.com dataset (formerly GeniusKitchen) introduced by Majumder et al. (EMNLP 2019):
• PP_users.csv: 25,076 users with techniques, ingredients tried, and rating counts
• PP_recipes.csv: 178,265 recipes with tokenized names, ingredients, steps, and calorie levels
• RAW_recipes.csv: 231,637 recipes with preparation time in minutes, tags, nutrition, and steps
• RAW_interactions.csv: 1,132,367 user ratings on a scale of 0 to 5 and textual reviews
• Ingr_map.pkl: 11,659 raw ingredients mapped to normalized ingredient categories

## Key Exploratory Data Analysis Insights

• User Rating Activity: The median number of ratings per user is 6. Approximately 73 percent of users rated only 1 recipe, while only 2.3 percent rated more than 25 recipes.
• Rating Score Distribution: Ratings are heavily skewed toward positive feedback. 72 percent of all reviews are 5 star ratings. Ratings of 0 represent user observations or bookmarks rather than true low scores, and were filtered out.
• Recipe Complexity: Most recipes take between 20 and 65 minutes to prepare, with 6 to 12 ingredients and 6 to 11 steps.
• Meal Types: Recipe tags map predominantly to main dish (32 percent), desserts (20 percent), and side dishes (11 percent).

## Recommendation Methodologies and Results

### 1. User and Item Based Collaborative Filtering

Multiple models from the Scikit Surprise package were benchmarked across 5 fold cross validation:
• Item based KNNBasic achieved the lowest MAE (0.3463) among default neighborhood models.
• GridSearch optimization for Item based KNNBasic yielded an optimal MAE of 0.3548 using Mean Squared Difference (MSD) and 13 neighbors.
• Matrix Factorization models showed strong performance: SVD (biased=True) achieved an MAE of 0.3526, and SVDpp achieved an MAE of 0.3481.
• GridSearch optimization for SVD achieved an MAE of 0.3491 with 50 latent factors, 20 epochs, learning rate 0.005, and regularization 0.02.

### 2. Cluster Based Collaborative Filtering

To address matrix sparsity and computation time, clustering approaches were investigated:
• CoClustering Algorithm: Evaluated across user clusters (1 to 9) and recipe clusters (2 to 20). Best cluster sizes (9 user clusters, 5 recipe clusters) produced MAE 0.3655. GridSearch best parameters (2 user clusters, 2 recipe clusters) produced MAE 0.3513.
• Determining User and Item Clusters: Scikit Learn Euclidean KMeans yielded poor cluster separation. NLTK Cosine KMeansClusterer provided much clearer elbow curves and silhouette coefficient distributions.
• Cluster Based Neighborhood Models: 16 models were evaluated across user and recipe clusters. User based KNNWithZScore achieved the lowest MAE of 0.3451.
• Qualitative Recommendation Audit (User 173579): SVD recommended universally popular dishes (pot roast, strawberry shortcake, bread), whereas the Cluster Based model suggested more diverse, personalized items (bacon parmesan brussels sprouts, broiled scallops, sausage gravy).

### 3. Content Based Recommendation System

A personalized content based recommender was developed using TF IDF vectors from recipe text descriptions:
• User profiles were constructed from recipes rated 4 or 5 stars by the user.
• Cosine similarity was computed between candidate recipes and user profile vectors to generate Top N recommendations.
• Quantitative Evaluation: Precision and Recall were evaluated for K values from 1.0 to 10.0 across test users. Precision peaked at K=2 (0.000708) and K=7 (0.000607), while Recall reached 0.004249 for K between 7 and 10.

## Key Conclusions

1. Content Based Model Limitations:
Textual recipe descriptions alone do not fully capture recipe appeal, leading to modest precision and recall metrics.
2. Cluster Based Personalization:
Cluster based collaborative filtering generated more personalized and diverse recommendations compared to global models by restricting similarity calculations to tightly related user sub communities.
3. Rating Imbalance:
Collaborative filtering models on Food.com achieve low error metrics (MAE approximately 0.35), but 72 percent 5 star rating prevalence causes pure collaborative models to exhibit popularity bias without semantic content filtering.

## Project Structure

• Recipe_Recommender.ipynb: Full Jupyter Notebook with data ingestion, EDA, collaborative filtering, clustering, and content based pipelines with executed outputs.
• Recipe_Recommender_Presentation.pptx: Research presentation slide deck with 13 slides matching the project report.
• data: Folder containing Food.com datasets.
• README.md: Project documentation and summary report.
