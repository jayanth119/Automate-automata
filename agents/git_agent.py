from github import Github
import os
import json

# Setup
from dotenv import load_dotenv
load_dotenv()
g = Github(add)
REPO_NAME = "openai/openai-cs-agents-demo" 
repo = g.get_repo(REPO_NAME)

# Save previous PR numbers (in local file)
STATE_FILE = "pr_state.json"

def load_previous_prs():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return []

def save_current_prs(pr_numbers):
    with open(STATE_FILE, "w") as f:
        json.dump(pr_numbers, f)

# Step 1: Load existing PR numbers
previous_prs = load_previous_prs()

# Step 2: Get current open PRs
open_prs = list(repo.get_pulls(state='open'))
current_pr_numbers = [pr.number for pr in open_prs]

# Step 3: Detect new PRs
new_prs = [pr for pr in open_prs if pr.number not in previous_prs]

if new_prs:
    print(f"🆕 Detected {len(new_prs)} new PR(s)")
    for pr in new_prs:
        print(f"\n--- New PR #{pr.number}: {pr.title} by {pr.user.login} ---")
        files = pr.get_files()
        for file in files:
            print(f"  📄 File: {file.filename}")
            print(f"    ➕ Additions: {file.additions}")
            print(f"    ➖ Deletions: {file.deletions}")
            print(f"    🔁 Changes: {file.changes}")
            print(f"    Status: {file.status}")
else:
    print("✅ No new pull requests found.")

save_current_prs(current_pr_numbers)
