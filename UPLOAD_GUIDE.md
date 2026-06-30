# 🛡️ Secure Upload Guide — GitCraft AI

## Before Uploading Files to This Tool

This AI tool analyses your code. To protect yourself:

### 1️⃣ Check for Hardcoded Secrets
Before uploading, scan your files:
```bash
grep -rn "api_key\|password\|token\|secret\|sk-\|gsk_\|AKIA" your_file.py
```

### 2️⃣ Replace Secrets with ENV Variables
```python
# ❌ Never do this
GROQ_API_KEY = "gsk_abc123..."

# ✅ Do this instead
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
```

### 3️⃣ Use python-dotenv Locally
```bash
# Create .env (never commit this file)
echo "GROQ_API_KEY=your_key" >> .env
echo ".env" >> .gitignore
```

### 4️⃣ Strip Notebook Outputs
If uploading a .ipynb that was previously run:
```bash
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

### 5️⃣ When Pushing to GitHub
Run a final check:
```bash
git diff --staged | grep -E "gsk_|sk-|ghp_|AIza|AKIA"
# If anything shows — DO NOT push. Fix it first.
```

### 6️⃣ Emergency: Key Was Pushed
1. Revoke the key at provider console IMMEDIATELY
2. Generate a new key
3. Use BFG Repo-Cleaner to purge from git history:
   `bfg --replace-text secrets.txt my-repo.git`
4. Force push the cleaned history
