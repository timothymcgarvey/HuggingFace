import requests
from datasets import load_dataset

import os
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
issues_dataset = load_dataset("json", data_files="datasets-issues-clean.jsonl", split="train")
print(issues_dataset.num_rows)
sample = issues_dataset.shuffle(seed=666).select(range(3))

# Print out the URL and pull request entries
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")

issues_dataset_ipr = issues_dataset.map(
    lambda x: {"is_pull_request": x["pull_request"] not in [None, "None"]}
)

print(issues_dataset_ipr.num_rows)
sample = issues_dataset_ipr.shuffle(seed=666).select(range(3))
for url, ipr in zip(sample["html_url"], sample["is_pull_request"]):
    print(f">> URL: {url}")
    print(f">> Is pull request: {ipr}\n")


issues_dataset_pure = issues_dataset_ipr.filter(lambda x: x["is_pull_request"] == True)
print(issues_dataset_pure.num_rows)



from datetime import datetime, timezone

def to_dt_utc(val):
    """Return a timezone-aware (UTC) datetime or None."""
    if val in (None, "None"):
        return None
    # Already a datetime?
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    # Parse string
    s = str(val).strip()
    # Try ISO first (handles 'YYYY-mm-dd HH:MM:SS[.us][+HH:MM]')
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    # Try common explicit formats
    for fmt in ("%Y-%m-%d %H:%M:%S%z",    # 2025-10-20 13:49:24+00:00
                "%Y-%m-%d %H:%M:%S",      # 2020-12-04 12:13:07
                "%Y-%m-%dT%H:%M:%S%z",    # 2020-12-05T15:40:18+00:00
                "%Y-%m-%dT%H:%M:%S"):     # 2020-12-05T15:40:18
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    # Handle trailing 'Z' (UTC) if present
    if s.endswith("Z"):
        try:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    # Couldn’t parse
    return None

def compute_time_to_resolve(example):
    created_dt = to_dt_utc(example.get("created_at"))
    closed_dt  = to_dt_utc(example.get("closed_at"))

    if created_dt is None or closed_dt is None:
        return {"time_to_resolve_issue": None}

    delta = closed_dt - created_dt
    return {"time_to_resolve_issue": delta.total_seconds() / 3600.0}  # hours

issues_dataset_DT = issues_dataset_pure.map(compute_time_to_resolve)





sample = issues_dataset_DT.shuffle(seed=42).select(range(5))
for c, cl, t in zip(sample["created_at"], sample["closed_at"], sample["time_to_resolve_issue"]):
    print("created:", c, " | closed:", cl, " | hours:", t)


issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)
print(response.json())

import os, json, time, requests
from pathlib import Path

CACHE_DIR = Path("issue_comments_cache")
CACHE_DIR.mkdir(exist_ok=True)
import requests, time, json
from pathlib import Path


def fetch_all_comments(owner, repo, headers, max_pages=1000):
    """
    Fetches ALL issue comments (not PR review comments) for a repo.
    """
    all_comments = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/comments"
        params = {"page": page, "per_page": 100}
        r = requests.get(url, headers=headers, params=params)

        # Rate limit handling
        if r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", 0))
            sleep_for = max(0, reset - time.time()) + 5
            print(f"Rate limit hit — sleeping for {sleep_for / 60:.1f} min")
            time.sleep(sleep_for)
            continue

        batch = r.json()
        if not batch:  # No more pages
            break

        all_comments.extend(batch)
        print(f"Fetched page {page}, total comments: {len(all_comments)}")
        page += 1
        time.sleep(0.3)

        if page > max_pages:
            print("Reached max_pages limit, stopping early.")
            break

    print(f"✅ Total comments fetched: {len(all_comments)}")
    return all_comments


# Example usage
comments = fetch_all_comments("huggingface", "datasets", headers)
Path("all_comments.json").write_text(json.dumps(comments, indent=2))

from collections import defaultdict
import re, json

issue_comments = defaultdict(list)

for c in comments:
    issue_url = c.get("issue_url", "")
    match = re.search(r"/issues/(\d+)", issue_url)
    if match:
        num = int(match.group(1))
        issue_comments[num].append(c.get("body", ""))


print(f"Issue comments collected for {len(issue_comments)} issues")
print(list(issue_comments.items())[:3])

issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": issue_comments.get(int(x["number"]), [])}
)

# Save as an arrow dataset (fastest)
issues_with_comments_dataset.save_to_disk("datasets_with_comments")

# or save as JSONL if you prefer plain text
issues_with_comments_dataset.to_json("datasets-with-comments.jsonl")


