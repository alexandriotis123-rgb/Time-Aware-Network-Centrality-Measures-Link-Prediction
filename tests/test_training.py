from prediction.training.brute_force import (
    generate_similarity_ranges,
    find_best_single_range,
    improve_range_set,
    train_similarity_measure
)

dataset = [

    # (u, v, GD, CN, JC, AA, PA, label)

    (1, 2, -1, 2, 0.10, 0.50, 10, 1),
    (1, 3, -2, 5, 0.35, 0.80, 20, 0),
    (2, 3, -1, 3, 0.60, 1.20, 15, 1),
    (2, 4, -3, 1, 0.75, 0.30, 25, 0),
    (3, 4, -2, 4, 0.90, 2.00, 30, 1),
]

candidate_edges = [

    (1,2),
    (1,3),
    (2,3),
    (2,4),
    (3,4)

]

ground_truth_edges = {

    (1,2),
    (2,3),
    (3,4)

}

print("\nTEST 1")

scores = [

    0.60,
    0.10,
    0.35,
    0.60

]

ranges = generate_similarity_ranges(scores)

for r in ranges:
    print(r)

print("\nTEST 2")

best_range, best_accuracy = find_best_single_range(

    dataset,
    candidate_edges,
    ground_truth_edges,
    "JC"

)

print(best_range)
print(best_accuracy)

print("\nTEST 3")

new_ranges, new_accuracy = improve_range_set(

    dataset,
    candidate_edges,
    ground_truth_edges,
    "JC",
    best_range,
    best_accuracy

)

print(new_ranges)
print(new_accuracy)

print("\nTEST 4")

ranges, accuracy = train_similarity_measure(

    dataset,
    candidate_edges,
    ground_truth_edges,
    "JC"

)

print(ranges)
print(accuracy)