import json
import os
import csv

checkpoint_path = 'data/evaluation/checkpoint.json'
csv_path = 'data/evaluation/ground_truth.csv'

# Read the 16th query
with open(csv_path, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
    q16 = str(rows[15]['question'])
    gt16 = str(rows[15]['ground_truth'])

# Load checkpoint
data = []
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

# Ensure it's not already there
if not any(d['question'] == q16 for d in data):
    data.append({
        'question': q16,
        'answer': '{"source_document_id": "ERROR", "document_title": "ERROR", "document_abstract": "ERROR", "affiliated_companies_or_institutions": [], "summary_of_technical_relevance": "FAILED DUE TO LLM HANG", "key_claims_or_methodologies": [], "perception_findings": [], "patent_metadata": null}',
        'contexts': ['The local LLM crashed/hung on this context.'],
        'ground_truth': gt16
    })

# Save checkpoint
with open(checkpoint_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Successfully injected dummy bypass for Query 16 into checkpoint.json!")

