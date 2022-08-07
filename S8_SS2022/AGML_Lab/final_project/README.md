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
random error: rmse = 1.785
random error: avg_miss = 1.422

user_based offline phase: avg time 51.7s
user_based online phase: avg time 3.05s
user_based error: rmse = 0.577
user_based error: avg_miss = 0.234

item_based offline phase: avg time 37.84s
item_based online phase: avg time 2.44s
item_based error: rmse = 0.569
item_based error: avg_miss = 0.222

cluster_users offline phase: avg time 16.73s
cluster_users online phase: avg time 3.18s
cluster_users error: rmse = 0.603
cluster_users error: avg_miss = 0.249

cluster_items offline phase: avg time 30.0s
cluster_items online phase: avg time 2.84s
cluster_items error: rmse = 0.670
cluster_items error: avg_miss = 0.301

als_factorization offline phase: avg time 16.15s
als_factorization online phase: avg time 0.21s
als_factorization error: rmse = 0.559
als_factorization error: avg_miss = 0.233

hybrid offline phase: avg time 155.96s
hybrid online phase: avg time 6.57s
hybrid error: rmse = 0.515
hybrid error: avg_miss = 0.203

timed all testing: 2615.36s
```
