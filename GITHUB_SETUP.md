# 🚀 Setting Up Your GitHub Repository

This guide will help you create a GitHub repository and push your code.

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website (Recommended)

1. **Go to GitHub**: https://github.com
2. **Sign in** to your account
3. **Click the "+" icon** (top right) → "New repository"
4. **Repository settings**:
   - **Name**: `deepscribe-evals` (or your preferred name)
   - **Description**: "Comprehensive evaluation framework for clinical SOAP note generation"
   - **Visibility**: Public (or Private if you prefer)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. **Click "Create repository"**

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI if not installed
# Windows: winget install GitHub.cli
# macOS: brew install gh
# Linux: See https://cli.github.com/

# Login to GitHub
gh auth login

# Create repository
gh repo create deepscribe-evals --public --description "Comprehensive evaluation framework for clinical SOAP note generation"
```

## Step 2: Initialize Git (If Not Already Done)

```bash
# Check if git is initialized
git status

# If not initialized, run:
git init
```

## Step 3: Add Remote Repository

After creating the GitHub repo, you'll see a URL like:
- `https://github.com/yourusername/deepscribe-evals.git` (HTTPS)
- `git@github.com:yourusername/deepscribe-evals.git` (SSH)

**Add the remote:**

```bash
# Replace with your actual repository URL
git remote add origin https://github.com/yourusername/deepscribe-evals.git

# Or if using SSH:
# git remote add origin git@github.com:yourusername/deepscribe-evals.git

# Verify remote was added
git remote -v
```

## Step 4: Stage and Commit Files

```bash
# Check what will be committed (should NOT show .env or results/*.json)
git status

# Add all files
git add .

# Commit
git commit -m "Initial submission: DeepScribe Evaluation Framework

- 3 evaluation pipelines (reference-based, non-reference, self-validation)
- Meta-analysis with Pearson correlations, Kappa, ICC
- Executive dashboard with 7+ visualizations
- Support for multiple LLM providers (Ollama, OpenAI, Gemini)
- Fast mode with pre-computed results
- Comprehensive documentation"
```

## Step 5: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

If you get authentication errors:
- **HTTPS**: GitHub will prompt for username/password (use Personal Access Token, not password)
- **SSH**: Make sure your SSH key is added to GitHub

## Step 6: Verify on GitHub

1. Go to your repository: `https://github.com/yourusername/deepscribe-evals`
2. Verify files are there
3. Check that README.md displays correctly
4. Verify `.env` is NOT visible (it's in .gitignore)

## Step 7: Create a Release (Optional but Recommended)

```bash
# Create a tag
git tag -a v1.0.0 -m "Initial submission version"

# Push the tag
git push origin v1.0.0
```

Then on GitHub:
1. Go to "Releases" → "Create a new release"
2. Select tag `v1.0.0`
3. Title: "Initial Submission"
4. Description: Copy from your README or add a summary
5. Click "Publish release"

## 🔐 Getting a Personal Access Token (For HTTPS)

If GitHub asks for authentication:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. **Name**: `deepscribe-evals-push`
4. **Expiration**: 90 days (or your preference)
5. **Scopes**: Check `repo` (full control of private repositories)
6. Click "Generate token"
7. **Copy the token** (you won't see it again!)
8. Use this token as your password when pushing

## ✅ Final Checklist

Before submitting the link:

- [ ] Repository is public (or shared with reviewers)
- [ ] README.md displays correctly on GitHub
- [ ] All files are committed (check `git status`)
- [ ] `.env` file is NOT visible (check `.gitignore`)
- [ ] `QUICK_START.md` is accessible
- [ ] Repository has a description
- [ ] Code is pushed to `main` branch

## 📋 Submission Template

When submitting to reviewers, use this format:

```
Repository: https://github.com/yourusername/deepscribe-evals

Quick Start: See QUICK_START.md for 5-minute setup guide

Key Features:
- 3 evaluation pipelines (reference-based, non-reference, self-validation)
- Meta-analysis with framework validation metrics
- Executive dashboard with 7+ visualizations
- Support for multiple LLM providers (Ollama, OpenAI, Gemini)
- Fast mode: Use --use-processed flag for instant results

Requirements:
- Python 3.8+
- Gemini API key (get from https://makersuite.google.com/app/apikey)

Setup Time: ~5 minutes
Execution Time: ~1 minute (fast mode)
```

## 🆘 Troubleshooting

**"Repository not found"**
- Check the repository URL is correct
- Verify repository is public (or you've shared access)

**"Authentication failed"**
- Use Personal Access Token instead of password
- Or set up SSH keys

**"Large file" error**
- Check for files > 100MB
- Remove them or use Git LFS

**"Nothing to commit"**
- Check `git status`
- Files might already be committed
- Try `git add -A` then `git commit`

## 🎉 You're Done!

Once pushed, your repository is live and ready for submission!

