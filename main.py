import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import hdbscan

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def cluster_data(file_path, is_public_data=True):
    """
    Loads data from a CSV file, performs clustering, and returns a DataFrame
    with original IDs and assigned cluster labels.

    Args:
        file_path (str): The path to the CSV file.
        is_public_data (bool): Flag to determine if it's the public dataset (n=4)
                               or private dataset (n=6).

    Returns:
        pandas.DataFrame: DataFrame with 'id' and 'cluster_label' columns,
                          or None if an error occurs.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if 'id' not in data.columns:
        print(f"Error: 'id' column not found in {file_path}.")
        return None

    ids = data['id']
    
    features = data.drop('id', axis=1)
    
    n_dimensions = features.shape[1]
    
    n_clusters = 4 * n_dimensions - 1
    
    print(f"Processing {file_path}:")
    print(f"  Number of dimensions (n): {n_dimensions}")
    print(f"  Calculated number of clusters (4n - 1): {n_clusters}")

    if features.empty:
        print(f"Error: No features found for clustering in {file_path} after dropping 'id'.")
        return None
        
    scaler_minmax = MinMaxScaler() # 數值的範圍是固定的
    scaled_features_minmax = scaler_minmax.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', init='k-means++', random_state=42) 

    labels = kmeans.fit_predict(scaled_features_minmax)

    results_df = pd.DataFrame({'id': ids, 'label': labels})
    
    return results_df


public_data_file = 'public_data.csv'
private_data_file = 'private_data.csv'

    
    
public_results_df = cluster_data(public_data_file, is_public_data=True)
if public_results_df is not None:
    public_output_file = 'public_submission.csv'
    public_results_df.to_csv(public_output_file, index=False)
    print(f"Cluster assignments for public data saved to {public_output_file}")


private_results_df = cluster_data(private_data_file, is_public_data=False)
if private_results_df is not None:
    private_output_file = 'private_submission.csv'
    private_results_df.to_csv(private_output_file, index=False)
    print(f"\nCluster assignments for private data saved to {private_output_file}")
