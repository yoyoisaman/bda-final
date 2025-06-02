# 2025 NTU BDA Final Project - Clustering Automation with Multiple Algorithms

## Introduction

The task is to analyze the relationships within this dataset and classify the data into several distinct categories. It is known that if the data has n dimensions, you should be able to clearly observe 4n – 1 clusters. 
This repo is the clustering process for both public and private datasets using the KMeans algorithm.
The script reads a CSV file containing data with an id column, scales the feature values, and assigns cluster labels based on a computed number of clusters (4n - 1, where n is the number of features). The results are then saved to output CSV files.

## Usage

To run the script, ensure the input files public_data.csv and private_data.csv are located in the same directory. Then execute:

```bash
python main.py
