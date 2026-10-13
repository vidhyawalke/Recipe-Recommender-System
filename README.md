# Recipe Recommendation System on Food.com
### Collaborative Filtering, Unsupervised Clustering, and Content Based Methods

**Researcher: Vidhya Walke**  
**Degree: MSc Data Science**  

Dataset Benchmark: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

## Summary

Online recipe platforms host hundreds of thousands of candidate meals. Everyday cooks face severe choice paralysis when searching for new recipes that match their culinary tastes, time constraints, and ingredient preferences. In this research project, I designed, implemented, and empirically evaluated an end to end recipe recommendation engine using the Food.com benchmark dataset consisting of 180K recipes and 1.13M user interactions.

I systematically investigated three distinct recommendation paradigms:
1. Classical Collaborative Filtering (User and Item Neighborhood models, SVD, SVDpp, and NMF via Scikit Surprise)
2. Cluster Based Collaborative Filtering (Matrix subspace partitioning via Surprise CoClustering, Euclidean KMeans, and NLTK Cosine KMeans)
3. Personalized Content Based Recommendations (TF IDF token representations derived from recipe descriptions, evaluated across Precision and Recall for K from 1 to 10)

Through rigorous 5 fold cross validation, grid search parameter optimization, and qualitative recommendation audits, I analyzed how catalog sparsity and severe rating imbalance impact recommendation quality.

## Core Learnings and Engineering Takeaways

Completing this research deepened my mastery of recommender systems in real world, highly skewed domains. The four most critical data science insights I learned are:

### 1. Low Prediction Error Does Not Equal Practical Personalization
During my evaluation of Scikit Surprise models, SVD achieved an outstanding Mean Absolute Error of 0.3491. However, when I conducted a qualitative audit on target users (such as User 173579), SVD repeatedly recommended identical, universally popular items (whole wheat bread, pumpkin bread, pot roast) across different user profiles. Because 72 percent of ratings in the Food.com dataset are 5 star reviews, unconstrained matrix factorization converges toward the global rating prior. I realized that optimizing solely for rating prediction error encourages popularity bias at the direct expense of personal discovery.

### 2. Distance Metric Selection Determines Clustering Viability in Sparse Spaces
When attempting to cluster users and recipes, standard Euclidean KMeans failed to provide meaningful cluster separation. Because user interaction vectors are extremely sparse and high dimensional, Euclidean distance degrades due to the curse of dimensionality, creating distorted cluster centroids. By implementing NLTK Cosine KMeansClusterer, I measured angular orientation rather than vector magnitude. This shift yielded distinct elbow curves and well separated silhouette distributions, proving that cosine similarity is essential when clustering sparse behavioral interaction matrices.

### 3. Subspace Neighborhood Isolation Solves Both Sparsity and Scalability
By partitioning the global interaction matrix into cohesive culinary taste clusters prior to neighbor calculation, I successfully eliminated global sparsity noise. Rather than scanning thousands of distant users, the neighborhood model calculates similarities strictly within the target user specific culinary cluster. My cluster based KNNWithZScore model attained the lowest overall MAE of 0.3451 and recommended diverse, specialized dishes (broiled scallops, calzone bread machine, homemade sausage gravy) that aligned with individual taste profiles.

### 4. Textual Recipe Descriptions Suffer From Narrative Vocabulary Dilution
My evaluation of the content based recommender revealed modest precision and recall metrics. By inspecting raw user submissions, I discovered that culinary descriptions provided by home cooks are heavily dominated by conversational anecdotes, family memories, and emotional stories rather than standardized ingredient combinations or flavor keywords. This narrative noise dilutes TF IDF vector representations, demonstrating that production grade culinary systems require structured ontology tagging rather than uncurated free text.

## Dataset Architecture

The research utilized the Food.com culinary benchmark introduced by Majumder et al. (EMNLP 2019). The data collection encompasses five interconnected relational structures:

• **PP_users.csv** (25,076 users): Contains processed user histories including technique tokens, ingredients explored, and total rating counts.  
• **PP_recipes.csv** (178,265 recipes): Contains preprocessed recipes with tokenized titles, normalized ingredient sequences, step sequences, and nutritional calorie brackets.  
• **RAW_recipes.csv** (231,637 recipes): Contains original recipe records including culinary metadata, preparation time in minutes, category tags, step descriptions, and ingredient lists.  
• **RAW_interactions.csv** (1,132,367 interactions): Contains user reviews and explicit ratings on a 0 to 5 scale spanning sixteen years of platform activity.  
• **Ingr_map.pkl** (11,659 items): Serialized replacement dictionary mapping raw ingredient text variations into standardized canonical ingredient identifiers.

### Data Cleaning and Preprocessing Pipeline
1. **Filtering Zero Ratings:** The dataset contains explicit ratings from 0 to 5. Exploratory analysis confirmed that ratings of 0 signify private bookmarks or non evaluative cooking notes rather than actual low scores. To ensure valid feedback signals, I filtered all 0 ratings, retaining explicit scores from 1 to 5.
2. **Missing Value Resolution:** Missing description fields and recipe names in raw records were resolved with string sanitization to prevent pipeline crashes during tokenization.
3. **Meal Tag Classification:** I processed multi valued tag arrays to classify recipes into twenty distinct meal categories, with main dish (32 percent), desserts (20 percent), and side dishes (11 percent) representing the dominant subsets.

## Exploratory Data Analysis Insights

My exploratory data analysis revealed several distinctive characteristics of culinary user behavior:

• **User Activity Sparsity:** The median number of ratings per user is only 6. Over 73 percent of users rated only 1 recipe, while only 2.3 percent of power users submitted more than 25 ratings. This intense long tail distribution creates extreme interaction sparsity.  
• **Severe Rating Imbalance:** Explicit ratings are overwhelmingly positive. 72 percent of all interactions are 5 star ratings, and 16.8 percent are 4 star ratings. Less than 4 percent of ratings represent negative feedback (1 or 2 stars).  
• **Recipe Complexity Profiles:** The majority of recipes require between 20 and 65 minutes of cooking time, utilizing 6 to 12 ingredients and 6 to 11 preparation steps.

## Recommendation Methodologies and Empirical Results

### 1. Classical Collaborative Filtering Benchmarks

I evaluated memory based neighborhood models and latent factor matrix factorization models using Scikit Surprise across identical 5 fold cross validation splits:

• **Item based KNNBasic:** Achieved the lowest error among default neighborhood algorithms with an MAE of 0.3463 and an RMSE of 0.5843.  
• **Neighborhood Hyperparameter Tuning:** Systematic GridSearch on Item KNNBasic revealed that Mean Squared Difference (MSD) similarity paired with k = 13 neighbors yielded optimal neighborhood stability with an MAE of 0.3548.  
• **Matrix Factorization:** SVD with user and item biases achieved an MAE of 0.3526, while SVDpp incorporating implicit feedback achieved an MAE of 0.3481.  
• **SVD GridSearch Optimization:** Hyperparameter tuning over latent dimension sizes, learning rates, and regularization penalties identified that 50 factors, 20 epochs, learning rate 0.005, and regularization 0.02 minimized error to an MAE of 0.3491.

### 2. Cluster Based Collaborative Filtering

To alleviate sparsity and prevent popularity collapse, I formulated a two stage clustering pipeline:

• **CoClustering Joint Optimization:** Simultaneous clustering of users and items across 5 fold cross validation indicated that 9 user clusters and 5 recipe clusters achieved an MAE of 0.3655. GridSearch over cluster parameters confirmed an optimal MAE of 0.3513 with 2 user clusters and 2 recipe clusters.  
• **NLTK Cosine KMeans Partitioning:** Applying Cosine distance clustering produced clear elbow inflections and well defined silhouette distributions, successfully isolating culinary user sub communities.  
• **Subspace Neighborhood Benchmarking:** I evaluated 16 distinct model configurations across user and recipe clusters. User based KNNWithZScore achieved the strongest overall performance across the entire study with an MAE of 0.3451, proving that local community normalization outperforms global calculation.

### 3. Qualitative Audit: SVD Popularity Bias vs Cluster Personalization

To inspect real world utility beyond abstract error metrics, I examined Top 10 recommendations generated for User 173579:

• **Global SVD Predictions:** 50 whole wheat bread, melt in your mouth pumpkin bread, key west lime cake, alton brown mac n cheese, ultimate lemon meringue pie. All predicted ratings fell between 4.89 and 4.94. The recommendations consisted exclusively of high volume platform favorites.  
• **Cluster Based Recommendations:** Bacon parmesan brussels sprouts, baklava salmon with honey butter walnuts, calzone bread machine, country sausage gravy, parmesan crusted broiled scallops. The recommendations exhibited rich culinary variety and targeted specialization.

### 4. Content Based Recommender and Precision Recall Evaluation

I constructed a content based recommendation engine using TF IDF token vectors derived from recipe text descriptions:

• **User Profile Synthesis:** For each target user, I aggregated TF IDF representations of recipes they had previously rated 4 or 5 stars into a composite preference vector.  
• **Candidate Ranking:** Cosine similarity was calculated between candidate recipe vectors and user preference vectors to generate Top N recommendations.  
• **Quantitative Precision and Recall Evaluation:** Across K values from 1.0 to 10.0, Precision peaked at K = 2 (0.000708) and K = 7 (0.000607), while Recall expanded monotonically from 0.001416 to 0.004249 at K = 7 through 10.

## Research Conclusions and Future Work

My experimental findings establish three core conclusions:
1. **Textual Feature Limitations:** Content based models relying on uncurated recipe descriptions struggle with precision and recall due to conversational narrative dilution. Incorporating structured ingredient ontologies and nutritional embeddings is necessary for content fidelity.
2. **Superiority of Cluster Partitioning:** Restricting neighborhood collaborative filtering to cosine clustered user communities provides the best trade off between low error (MAE 0.3451) and recommendation novelty.
3. **Addressing Rating Imbalance:** Because 72 percent of user feedback consists of 5 star reviews, unconstrained matrix factorization suffers from systemic popularity bias.

### Next Generation Architecture
As future work, I propose a two stage hybrid recommendation framework:
• **Stage 1 (Subspace Candidate Retrieval):** Utilize Cosine Cluster Based Collaborative Filtering to retrieve top candidate recipes reflecting community preferences.  
• **Stage 2 (Semantic Ingredient Re Ranking):** Re rank retrieved candidates using structured ingredient graphs, preparation time constraints, and explicit user allergen filters to deliver production ready culinary personalization.

## Repository Structure

• **Recipe_Recommender.ipynb:** Primary Jupyter Notebook containing data loading, preprocessing, exploratory visualizations, Scikit Surprise benchmarks, NLTK Cosine KMeans clustering, and Content Based evaluation with fully executed outputs.  
• **README.md:** Complete project documentation and research summary report.  
• **data/:** Directory housing the Food.com benchmark datasets.

## Technical Stack and Environment

• **Programming Language:** Python 3.8+  
• **Recommender Systems:** Scikit Surprise (KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, CoClustering)  
• **Natural Language Processing & Clustering:** NLTK (KMeansClusterer, cosine_distance), Scikit Learn (TfidfVectorizer, cosine_similarity, KMeans, Silhouette analysis)  
• **Data Analysis & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
