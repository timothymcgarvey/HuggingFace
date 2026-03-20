






from datasets import load_dataset
issues_with_comments_dataset = load_dataset("json", data_files="datasets-with-comments.jsonl")
issues_with_comments_dataset.push_to_hub("github-issues")