"""
Construction of feature vectors for candidate edges.
"""

from src.analysis.similarity import (
    shortest_path_similarity,
    common_neighbors_similarity,
    jaccard_similarity,
    adamic_adar_similarity,
    preferential_attachment_similarity,)


def build_feature_vectors(graph,candidate_edges):
    """
    Generate feature vectors for candidate edges.
    """

    for node_u, node_v in candidate_edges:

        sp = shortest_path_similarity(graph,node_u,node_v)

        cn = common_neighbors_similarity(graph,node_u,node_v)

        jc = jaccard_similarity(graph,node_u,node_v)

        aa = adamic_adar_similarity(graph,node_u,node_v)

        pa = preferential_attachment_similarity(graph,node_u,node_v)

        yield (node_u,node_v,sp,cn,jc,aa,pa)