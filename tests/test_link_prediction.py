from src.prediction.link_prediction import predict_edges


dataset = [
    # (u, v, GD, CN, JC, AA, PA, label)

    (1, 2, -1, 2, 0.10, 0.50, 10, 1),
    (1, 3, -2, 5, 0.35, 0.80, 20, 0),
    (2, 3, -1, 3, 0.60, 1.20, 15, 1),
    (2, 4, -3, 1, 0.75, 0.30, 25, 0),
    (3, 4, -2, 4, 0.90, 2.00, 30, 1),]

predicted_edges = predict_edges(dataset, "JC", [(0.00, 1.00)])

print(predicted_edges)