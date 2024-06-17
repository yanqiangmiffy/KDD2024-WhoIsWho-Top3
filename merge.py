import json
from tqdm import tqdm


def normalize_scores(data):
    """Normalize the scores for each author-paper pair within each author's dictionary."""
    for author_id, papers in tqdm(data.items()):
        max_score = max(papers.values())
        min_score = min(papers.values())
        for paper_id in papers:
            # Normalize scores to a 0-1 range
            papers[paper_id] = (papers[paper_id] - min_score) / (max_score - min_score) if max_score != min_score else 0
    return data


def merge_results(weight1, data1, weight2, data2):
    """Merge two dictionaries of normalized scores, applying the given weights."""
    merged_results = {}
    for author_id in data1:
        merged_results[author_id] = {}
        papers1 = data1.get(author_id, {})
        papers2 = data2.get(author_id, {})
        all_papers = set(papers1.keys()).union(set(papers2.keys()))
        for paper_id in all_papers:
            score1 = papers1.get(paper_id, 0)
            score2 = papers2.get(paper_id, 0)
            # Weighted average of scores
            merged_results[author_id][paper_id] = weight1 * score1 + weight2 * score2
    return merged_results


# Load the data from the JSON files
with open('chtglm3的预测结果路径', 'r') as f:
    result_chatglm3 = json.load(f)

with open('glm4的预测结果路径', 'r') as f:
    result_glm4 = json.load(f)

with open('mistral的预测结果路径', 'r') as f:
    result_mistral = json.load(f)

with open('lgb1的预测结果路径', 'r') as f:
    result_lgb = json.load(f)

with open('lgb2的预测结果路径', 'r') as f:
    result_lgb4 = json.load(f)

with open('lgb3的预测结果路径.json', 'r') as f: 
    result_lgb3 = json.load(f)


# Normalize the scores
normalized_chatglm3= normalize_scores(result_chatglm3)
normalized_glm4 = normalize_scores(result_glm4)
normalized_mistral = normalize_scores(result_mistral)
normalized_lgb = normalize_scores(result_lgb)
normalized_lgb3 = normalize_scores(result_lgb3)
normalized_lgb4 = normalize_scores(result_lgb4)

# Merge the results with specified weights
# merged_data = merge_results(0.55, normalized_data1, 0.45, normalized_data2)

merged_results = {}
for author_id in result_chatglm3:
    merged_results[author_id] = {}
    papers1 = normalized_chatglm3.get(author_id, {})
    papers2 = normalized_glm4.get(author_id, {})
    papers4 = normalized_mistral.get(author_id, {})
    papers5 = normalized_lgb.get(author_id, {})
    papers8 = normalized_lgb3.get(author_id, {})
    papers9 = normalized_lgb4.get(author_id, {})
    all_papers = set(papers1.keys()).union(set(papers2.keys()))
    for paper_id in all_papers:
        score1 = papers1.get(paper_id, 0) #chatglm3
        score2 = papers2.get(paper_id, 0) #glm4
        score4 = papers4.get(paper_id, 0) #mistral
        score5 = papers5.get(paper_id, 0) #lgb1
        score8 = papers8.get(paper_id, 0) #lgb3
        score9 = papers9.get(paper_id, 0) #lgb2
        score_ = score5 * 0.2 + + score8 * 0.2 + score4 * 0.2 + score2 * 0.2 + score1 * 0.2
        merged_results[author_id][paper_id] = score_ * 0.8 + score9 * 0.2
# Save the merged results to a new JSON file
with open('final_result.json', 'w') as f_out:
    json.dump(merged_results, f_out, indent=4)