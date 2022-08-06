# Final Project: Recommender Systems

Here, Maurice Wenig is trying to implement a good Recommender System for the final project of AGML (Statistical Learning Theory) Lab.

## Usage

Run main: `python src/main.py`. Testing all recommenders: `python src/testall.py`. Example results (rounded predictions):

```text
mean offline phase: avg time 0.0s
mean online phase: avg time 0.0s
mean error: rmse = 0.926
mean error: avg_miss = 0.395

random offline phase: avg time 0.0s
random online phase: avg time 0.0s
random error: rmse = 1.783
random error: avg_miss = 1.421

user_based offline phase: avg time 61.88s
user_based online phase: avg time 3.63s
user_based error: rmse = 0.578
user_based error: avg_miss = 0.235

cluster_users offline phase: avg time 19.07s
cluster_users online phase: avg time 3.38s
cluster_users error: rmse = 0.606
cluster_users error: avg_miss = 0.251

item_based offline phase: avg time 41.31s
item_based online phase: avg time 2.43s
item_based error: rmse = 0.570
item_based error: avg_miss = 0.223

cluster_items offline phase: avg time 32.57s
cluster_items online phase: avg time 2.91s
cluster_items error: rmse = 0.667
cluster_items error: avg_miss = 0.300

als_factorization offline phase: avg time 17.32s
als_factorization online phase: avg time 0.21s
als_factorization error: rmse = 0.559
als_factorization error: avg_miss = 0.234

hybrid offline phase: avg time 145.24s
hybrid online phase: avg time 5.76s
hybrid error: rmse = 0.525
hybrid error: avg_miss = 0.203
```
