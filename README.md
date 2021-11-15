# interpretable_clustering
Interpretable Clustering: Voronoi Feature Importance 

This repository presents a general technique to explain what features are 
important to the division of cluster in cluster analysis. 
This feature importance technique uses Voronoi Diagrams to calculate feature
importance in n-dimensional case. This technique has been developed for 
existing partitioning and hierarchical clustering algorithms. 


This repository contains:
* Implementation of clustering algorithms: K-Means and Agglomerative Clustering 
(for the purpose of looking "under the hood")
* Feature Importance: General implementation of our feature importance technique,
adapted version for agglomerative clustering
* Synthetic Datasets: generated synthetic datasets with a ground truth feature importance
* Accuracy Testing: Calculating the accuracy of our feature importance to predict 
the ground truth of each synthetic dataset
* Stability Testing: Verify the additional of noise to a dataset doesn't dramatically 
effect feature importance (assuming the dataset remains consistent)


Feature Importance is calculated as follows:
* Adjacent pairs of clusters are defined as sharing a Voronoi Face
* A Voronoi Face is a decision boundary between two clusters and it's 
feature importance is the coefficients of that decision boundary normalised to sum to 1
(this is equivalent to the vector between the two sites that define the Voronoi Face) 
* The volume of each Voronoi Face is calculated as the amount contained in the bounding
box defined by the sites
* The total feature importance is calculated as a weighted average: 
```math
feature_importance = \frac{\sum_{d in decision_boundaries} v(d) * fi(d) }
{\sum_{d in decision_boundaries} 1}
```
where v(d) is the volume of decision boundary d in the bounding box 
and 
 fi(d) is the feature importance of decision boundary d

Repository currently being developed.

Feel free to contact me gem.campbell@hotmail.com