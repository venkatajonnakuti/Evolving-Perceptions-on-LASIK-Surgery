import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np

import re
from tqdm import tqdm

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.decomposition import PCA
from umap import UMAP
from umap import plot as umap_plot

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')


df = pd.read_excel('/mnt/belinda_local/venkata/data/Project_LASIK/Reddit CAC DB Searchstring.xlsx', index_col=0)

# Fill empty cells and remove some weird html tags
df['body'].fillna("", inplace=True)
df['body'] = df['body'].str.replace("http\S+", "")
df['body'] = df['body'].str.replace("\\n", " ")
df['body'] = df['body'].str.replace("&gt;", "")
df['body'] = df['body'].str.replace('\s+', ' ', regex=True)
df['body_len'] = df['body'].str.len()
df = df.query('body_len >= 25')

df = df.loc[df['body'].str.contains("(coronary artery calcium)|(coronary calcium)|(cac score)|(calcium score)|(calcium scan score)|(heart scan)", regex=True)]

# Turn into list
texts = df['body']
texts_list = texts.to_list()

# Calculate embeddings, so you don't have to do this every time you run the topic model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(texts_list, show_progress_bar=True)

# Assess for the optimal number of topics.
vectorizer_model = CountVectorizer(stop_words="english")
spec_topic_model = BERTopic(vectorizer_model=vectorizer_model)

n_topics = 0
n_iter = 3
for i in range(n_iter):
    spec_topics, _ = spec_topic_model.fit_transform(texts_list, embeddings)
    print(f'Number of topics in run {i} : {np.max(spec_topics)}')
    n_topics += np.max(spec_topics) / n_iter

n_topics = np.round(n_topics).astype('int')

print(n_topics)

cluster_model = SpectralClustering(n_clusters=n_topics, random_state=42)
umap_model = UMAP(n_neighbors=100, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
vectorizer_model = CountVectorizer(stop_words="english")
spec_topic_model = BERTopic(umap_model=umap_model, hdbscan_model=cluster_model, vectorizer_model=vectorizer_model)
spec_topics, _ = spec_topic_model.fit_transform(texts_list, embeddings)

spec_topic_model.save('/mnt/belinda_local/venkata/data/Project_LASIK/topic_model_save_new.npy')
spec_topic_model.visualize_hierarchy(orientation='bottom')

# print(texts_list)
# print(spec_topics)

hierarchical_topics = spec_topic_model.hierarchical_topics(texts_list)
tree = spec_topic_model.get_topic_tree(hierarchical_topics)
print(tree)

df['topic'] = spec_topics
kw = spec_topic_model.get_topic_info()['Name']
df['keywords'] = [kw.loc[i] for i in df['topic']]
df['topic'] += 1
df.to_excel('/mnt/belinda_local/venkata/data/Project_LASIK//cac_topics.xlsx')


from sklearn.preprocessing import MinMaxScaler
spec_topic_model = BERTopic.load('/mnt/belinda_local/venkata/data/Project_LASIK/topic_model_save_new.npy')

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from matplotlib import colormaps as cm

def find_clustering_scores(c_tf_idf_embed, llim=3, ulim=30):
    
    ss = []
    db = []
    
    cluster_arr = np.arange(llim, ulim)
    
    for n_clusters in cluster_arr:
        clusters = SpectralClustering(n_clusters=n_clusters, random_state=42, n_components=2).fit_predict(c_tf_idf_embed)
        ss.append(silhouette_score(c_tf_idf_embed, clusters))
        db.append(davies_bouldin_score(c_tf_idf_embed, clusters))
        
    with sns.plotting_context('notebook'):
        sns.set_style('ticks')
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 5))
        sns.lineplot(x=cluster_arr, y=ss, palette='autumn', ax=ax[0], color=cm['autumn'](0.3))
        sns.lineplot(x=cluster_arr, y=db, palette='autumn', ax=ax[1], color=cm['autumn'](0.7))
        
        ax[0].set_ylabel('Silhouette Score')
        ax[1].set_ylabel('Davies-Bouldin Score')
        
        ax[1].set_xlabel('Number of Clusters')
        
        fig.suptitle('Clustering Performance', fontsize=15, y=0.95)

    ideal_n_clusters = cluster_arr[np.argmax(ss)]

    print("top silhouette score: {0:0.3f} for at n_clusters {1}".format(np.max(ss), cluster_arr[np.argmax(ss)]))
    print("top davies-bouldin score: {0:0.3f} for at n_clusters {1}".format(np.min(db), cluster_arr[np.argmin(db)]))
    
    return ideal_n_clusters

c_tf_idf_mms = mms().fit_transform(spec_topic_model.c_tf_idf_.toarray())
c_tf_idf_embed_vis = UMAP(n_neighbors=2, n_components=2, metric='hellinger', random_state=42, spread=2, low_memory=False).fit_transform(c_tf_idf_mms)
c_tf_idf_embed = UMAP(n_neighbors=2, n_components=3, metric='hellinger', random_state=42, spread=2, low_memory=False).fit_transform(c_tf_idf_mms)

ideal_n_clusters = find_clustering_scores(c_tf_idf_embed)
c_tf_idf_embed_clust = KMeans(n_clusters=ideal_n_clusters, random_state=42).fit_predict(c_tf_idf_embed) + 1


with sns.plotting_context('notebook'):
    sns.set_style('white')
    plt.figure(figsize=(10, 10))

    # vis_arr = c_tf_idf_embed[:, [0, 2]]
    vis_arr = c_tf_idf_embed_vis

    ax = sns.scatterplot(x=vis_arr[:, 0], y=vis_arr[:, 1], size=spec_topic_model.get_topic_info()['Count'],
                         hue=c_tf_idf_embed_clust, 
                         sizes=(700, 2000),
                         alpha=0.6, palette='Paired', legend=True, edgecolor='k')
    h, l = ax.get_legend_handles_labels()
    for i, coords in enumerate(vis_arr):
        ax.annotate(i + 1, coords - [0.09, 0.05], fontsize=10)
    plt.legend(h[0:ideal_n_clusters], l[0:ideal_n_clusters])  # Adjust legend positioning or formatting as needed
    ax.set_title('Topics, Grouped by Similarity of Content', fontsize=16, pad=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Save the figure to a file in your local directory
    plt.savefig('/mnt/belinda_local/venkata/data/Project_LASIK/TopicGrouping.png', bbox_inches='tight')
    # Optionally, you can close the figure to free memory if you're generating many plots
    plt.close()



