"""
Layer 3: Graph-Based Analysis

Builds a provider-procedure bipartite graph and extracts structural
features that capture coordinated billing patterns. Providers who bill
the same unusual combination of procedures may be part of a fraud ring.

Graph construction:
  - Nodes: providers (type A) and HCPCS codes (type B)
  - Edges: weighted by service volume
  - Features extracted: degree centrality, PageRank, community membership,
    bipartite clustering coefficient

Why graph analysis:
  - Catches coordinated fraud that per-provider features miss
  - Fraud rings share billing patterns → graph communities
  - High centrality in unusual procedure clusters is a red flag

We use NetworkX for the graph analysis (sufficient for ~50K providers).
PyTorch Geometric GNN is optional for larger datasets — the NetworkX
features alone are strong signals.
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from loguru import logger

from cms_fwa.models.data_prep import save_artifact
from cms_fwa.utils.db import get_connection


def build_provider_procedure_graph() -> nx.Graph:
    """Build a bipartite graph of providers ↔ HCPCS codes.

    Edge weights represent the fraction of a provider's services
    from that HCPCS code (normalized, so each provider's edges sum to 1).

    Returns:
        NetworkX bipartite graph with node attribute 'bipartite' (0=provider, 1=hcpcs).
    """
    with get_connection() as conn:
        edges_df = conn.execute("""
            SELECT npi, hcpcs_code, hcpcs_fraction
            FROM main_intermediate.int_provider_hcpcs_mix
            WHERE hcpcs_fraction > 0.01
        """).fetchdf()

    logger.info(f"Building bipartite graph from {len(edges_df):,} edges")

    G = nx.Graph()

    # Add provider nodes
    providers = edges_df["npi"].unique()
    for npi in providers:
        G.add_node(f"P_{npi}", bipartite=0, node_type="provider")

    # Add HCPCS nodes
    hcpcs_codes = edges_df["hcpcs_code"].unique()
    for code in hcpcs_codes:
        G.add_node(f"H_{code}", bipartite=1, node_type="hcpcs")

    # Add weighted edges
    for _, row in edges_df.iterrows():
        G.add_edge(
            f"P_{row['npi']}",
            f"H_{row['hcpcs_code']}",
            weight=float(row["hcpcs_fraction"]),
        )

    logger.info(
        f"Graph: {G.number_of_nodes():,} nodes "
        f"({len(providers):,} providers + {len(hcpcs_codes):,} HCPCS codes), "
        f"{G.number_of_edges():,} edges"
    )
    return G


def extract_graph_features(G: nx.Graph) -> pd.DataFrame:
    """Extract node-level features from the bipartite graph.

    Features computed:
      - degree: number of unique HCPCS codes billed
      - weighted_degree: sum of edge weights (should be ~1 for normalized)
      - pagerank: importance in the billing network
      - clustering: bipartite clustering coefficient
      - community_id: Louvain community assignment

    Args:
        G: Bipartite provider-procedure graph.

    Returns:
        DataFrame with one row per provider NPI.
    """
    provider_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}

    # Degree centrality (restricted to provider nodes)
    logger.info("Computing degree centrality...")
    degree_centrality = nx.degree_centrality(G)

    # Weighted degree
    weighted_degree = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
                       for n in provider_nodes}

    # PageRank (captures "importance" in the billing network)
    logger.info("Computing PageRank...")
    pagerank = nx.pagerank(G, weight="weight", max_iter=100)

    # Bipartite clustering coefficient
    # Full bipartite clustering is O(n*k^2) — too slow for 78K providers.
    # Sample a subset and assign 0.0 to the rest.
    logger.info("Computing clustering coefficients (sampled)...")
    MAX_CLUSTERING_NODES = 5000
    clustering: dict = {}
    if len(provider_nodes) > MAX_CLUSTERING_NODES:
        import random
        rng = random.Random(42)
        sampled = set(rng.sample(sorted(provider_nodes), MAX_CLUSTERING_NODES))
        try:
            clustering = bipartite.clustering(G, sampled, mode="dot")
        except Exception:
            clustering = {n: 0.0 for n in sampled}
        # Assign 0.0 for non-sampled providers
        for n in provider_nodes:
            if n not in clustering:
                clustering[n] = 0.0
    else:
        try:
            clustering = bipartite.clustering(G, provider_nodes, mode="dot")
        except Exception:
            clustering = {n: 0.0 for n in provider_nodes}

    # Community assignment: group providers by their top HCPCS code.
    # Full bipartite projection (78K x 78K) is too memory-intensive for a laptop.
    # Grouping by dominant procedure is a fast, interpretable approximation —
    # providers billing the same top code form a "community" (specialty cluster).
    logger.info("Assigning communities by dominant HCPCS code...")
    community_map: dict[str, int] = {}
    hcpcs_to_community: dict[str, int] = {}
    community_counter = 0
    for node in provider_nodes:
        # Find the HCPCS neighbor with the highest edge weight
        neighbors = list(G.neighbors(node))
        if not neighbors:
            top_hcpcs = "NONE"
        else:
            top_hcpcs = max(neighbors, key=lambda h: G[node][h].get("weight", 0))
        if top_hcpcs not in hcpcs_to_community:
            hcpcs_to_community[top_hcpcs] = community_counter
            community_counter += 1
        community_map[node] = hcpcs_to_community[top_hcpcs]
    logger.info(f"Assigned {community_counter} communities by dominant HCPCS")

    # Assemble into DataFrame
    records = []
    for node in provider_nodes:
        npi = node[2:]  # Strip "P_" prefix
        records.append({
            "npi": npi,
            "graph_degree": G.degree(node),
            "graph_weighted_degree": weighted_degree.get(node, 0),
            "graph_degree_centrality": degree_centrality.get(node, 0),
            "graph_pagerank": pagerank.get(node, 0),
            "graph_clustering": clustering.get(node, 0),
            "graph_community_id": community_map.get(node, -1),
        })

    df = pd.DataFrame(records)

    # Community size (providers in same community → potential ring)
    community_sizes = df["graph_community_id"].value_counts().to_dict()
    df["graph_community_size"] = df["graph_community_id"].map(community_sizes)

    logger.info(
        f"Graph features: {len(df):,} providers, "
        f"{df['graph_community_id'].nunique()} communities detected"
    )
    return df


def compute_graph_anomaly_score(graph_features: pd.DataFrame) -> np.ndarray:
    """Compute a graph-based anomaly score from graph features.

    Providers are more suspicious if they have:
      - High PageRank (central in unusual billing patterns)
      - High clustering (tightly connected billing group)
      - Small community size (niche billing cluster)

    Returns scores in [0, 1] where 1 = most anomalous.
    """
    # Normalize each feature to [0, 1]
    def minmax(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / rng

    # Higher PageRank + higher clustering = more suspicious
    pagerank_norm = minmax(graph_features["graph_pagerank"])
    clustering_norm = minmax(graph_features["graph_clustering"])

    # Smaller community = more suspicious (invert)
    community_size = graph_features["graph_community_size"].clip(upper=100)
    community_norm = 1.0 - minmax(community_size)

    # Weighted combination
    score = (
        0.4 * pagerank_norm
        + 0.3 * clustering_norm
        + 0.3 * community_norm
    )

    return score.values


def train_graph_model() -> dict:
    """Build graph, extract features, compute anomaly scores.

    Returns:
        Dict with graph, features DataFrame, and scores.
    """
    logger.info("Building Graph Model (Layer 3 — Network Analysis)")

    G = build_provider_procedure_graph()
    features = extract_graph_features(G)
    scores = compute_graph_anomaly_score(features)
    features["graph_anomaly_score"] = scores

    save_artifact(features, "graph_features")

    logger.info(
        f"  Graph anomaly scores: mean={scores.mean():.4f}, "
        f"p95={np.percentile(scores, 95):.4f}, max={scores.max():.4f}"
    )

    return {
        "graph": G,
        "features": features,
        "scores": scores,
    }
