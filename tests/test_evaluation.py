from src.prediction.evaluation import (
    compute_lambda,
    compute_tpr,
    compute_tnr,
    compute_accuracy
)


candidate_edges = [
    (1,2),
    (1,3),
    (1,4),
    (2,3),
    (2,4),
    (3,4),
    (3,5),
    (4,5),
    (4,6),
    (5,6)
]

ground_truth_edges = {
    (1,2),
    (2,4),
    (3,4),
    (5,6)
}

predicted_edges = {
    (1,2),
    (1,3),
    (2,4),
    (3,5),
    (5,6)
}
print("Lambda:", compute_lambda(
    ground_truth_edges,
    candidate_edges
))

print("TPR:", compute_tpr(
    predicted_edges,
    ground_truth_edges
))

print("TNR:", compute_tnr(
    predicted_edges,
    ground_truth_edges,
    candidate_edges
))

print("Accuracy:", compute_accuracy(
    predicted_edges,
    ground_truth_edges,
    candidate_edges
))