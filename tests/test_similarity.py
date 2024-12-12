from run_experiment import evaluate_similarity

query = "I love dogs and cats"
response = "I love dogs and cats"
response2 = "I love fish and humans"

print(evaluate_similarity(query, response))
print(evaluate_similarity(query, response2))