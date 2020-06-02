from sklearn.metrics.pairwise import cosine_similarity

v1 = [0.0220125, 0.0596375, 0.0596375, 0.0220125, 0.0220125, 0.0596375,
      0.0220125, 0.0220125
      ] + [0] * 12
v2 = [0.019371] + [0] * 7 + [0.052481] * 8 + [0] * 4
v3 = [0] * 3 + [0.0220125] * 2 + [0] + [0.0220125] * 2 + [0] * 8 + [0.0596375] * 4
print(cosine_similarity([v1], [v3]))
print(cosine_similarity([v2], [v3]))