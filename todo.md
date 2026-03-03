# ML Challenge — Team Tech Ninjas: Competition Strategy & TODO

## Team Info
- **Team Name:** Tech Ninjas
- **Team Lead:** Rajat Pundir — Graphic Era Hill University, Dehradun | B.Tech CSE (AI & ML), 3rd Year
- **Member 2:** Sidh Khurana — Same class
- **Competition:** IEEE SB, GEHU | ML Challenge — Binary Fault Detection

---

## Understanding the Scoring Matrix

| Priority | Criteria | Weight | Current Status |
|----------|----------|--------|----------------|
| **1st** | **Accuracy** on hidden split | Primary | CV: 0.9878 — strong |
| **2nd** | **F1-Score** on hidden split | Primary | CV: 0.9848 — strong |
| **3rd (Tie-breaker)** | Code optimization | Tie-break | Needs improvement |
| **3rd (Tie-breaker)** | Architectural elegance | Tie-break | Needs improvement |
| **3rd (Tie-breaker)** | Overall code quality | Tie-break | Needs improvement |

> **F1/Accuracy are already excellent.** The battle will likely be won on **tie-breaker criteria** — code quality, clean architecture, optimization. That's where proper work division and GitHub history matter.

---

## Competition-Optimized Work Division

Since GitHub is **auto-integrated**, judges see your repo directly — commit history, code structure, branch strategy, and who did what are all visible.

### Rajat Pundir — Team Lead
**Role: Architecture + Modeling + Integration**

| Task | Deliverable | Why It Matters for Scoring |
|------|-------------|---------------------------|
| Project architecture | Clean folder structure, `config.py`, modular design | Tie-breaker: "architectural elegance" |
| Feature engineering pipeline | `src/feature_engineering.py` | Shows systematic approach, clean code |
| Model training + selection | `src/model.py` | Core scoring: best F1/Accuracy |
| Final pipeline orchestration | `src/pipeline.py` (single entry point) | Tie-breaker: "code optimization" |
| README + documentation | Professional `README.md` with results | Tie-breaker: "overall quality" |
| Code review (Sidh's PRs) | Review comments on GitHub | Shows collaboration, team quality |

### Sidh Khurana — Member 2
**Role: Data Engineering + Analysis + Validation**

| Task | Deliverable | Why It Matters for Scoring |
|------|-------------|---------------------------|
| Exploratory Data Analysis | `notebooks/eda.ipynb` | Shows rigorous analysis before modeling |
| Preprocessing pipeline | `src/preprocessing.py` (convert notebook → clean module) | Tie-breaker: "code optimization" |
| Data validation & testing | `tests/test_pipeline.py` | Tie-breaker: "overall quality" |
| Visualization of results | `notebooks/results_analysis.ipynb` | Shows understanding, not just running code |
| Requirements & environment | `requirements.txt`, `.gitignore` | Tie-breaker: "overall quality" |

---

## Recommended Repo Structure (Architectural Elegance)

```
ml-challenge-ieee/
├── README.md                          ← Rajat
├── requirements.txt                   ← Sidh
├── .gitignore                         ← Sidh
├── config.py                          ← Rajat (all hyperparams, paths)
├── run.py                             ← Rajat (single entry: python run.py)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               ← Sidh (modular preprocessing)
│   ├── feature_engineering.py          ← Rajat (47 engineered features)
│   └── model.py                        ← Rajat (LightGBM training + CV)
│
├── notebooks/
│   ├── 01_eda.ipynb                    ← Sidh (exploratory analysis)
│   ├── 02_preprocessing.ipynb          ← Sidh (preprocessing walkthrough)
│   └── 03_results_analysis.ipynb       ← Sidh (model performance viz)
│
├── data/
│   ├── raw/
│   │   ├── TRAIN.csv
│   │   └── TEST.csv
│   └── processed/
│       ├── TRAIN_PREPROCESSED.csv
│       └── TEST_PREPROCESSED.csv
│
├── output/
│   └── FINAL.csv                       ← Generated submission
│
└── tests/
    └── test_pipeline.py                ← Sidh (basic validation)
```

---

## GitHub Submission Strategy (Step-by-Step)

### Phase 1: Rajat creates repo & sets architecture

```powershell
# 1. Create repo on GitHub: "ml-challenge-ieee" (or whatever required)
# 2. Clone it locally
cd "D:\"
git clone https://github.com/RAJAT_USERNAME/ml-challenge-ieee.git
cd ml-challenge-ieee

# 3. Create folder structure
mkdir src, notebooks, data, data\raw, data\processed, output, tests

# 4. Create essential files
echo "" > src\__init__.py
echo "" > config.py
echo "" > run.py

# 5. Add .gitignore
# Add: __pycache__/, *.pyc, .ipynb_checkpoints/, *.egg-info/, .env, data/raw/*.csv

# 6. Add requirements.txt
# Add: numpy>=1.24.0, pandas>=2.0.0, scikit-learn>=1.3.0, xgboost>=2.0.0,
#       lightgbm>=4.0.0, matplotlib>=3.7.0, seaborn>=0.12.0, scipy>=1.11.0

# 7. Initial commit
git add .
git commit -m "chore: initialize project structure and configuration"
git push origin main
```

### Phase 2: Add Sidh as Collaborator
1. Go to **GitHub repo** → **Settings** → **Collaborators**
2. Click **Add people** → search Sidh's GitHub username → **Add**
3. Sidh accepts the invite from email/GitHub notifications

### Phase 3: Both work on branches simultaneously

**Sidh's workflow:**
```powershell
git clone https://github.com/RAJAT_USERNAME/ml-challenge-ieee.git
cd ml-challenge-ieee

# Branch 1: EDA
git checkout -b feat/eda-analysis
# Copy datacheck.ipynb → notebooks/01_eda.ipynb
git add notebooks/01_eda.ipynb
git commit -m "feat(eda): add exploratory data analysis with distribution and correlation study"
git push origin feat/eda-analysis
# → Create Pull Request on GitHub → Rajat reviews & merges

# Branch 2: Preprocessing
git checkout main; git pull
git checkout -b feat/preprocessing-pipeline
# Copy preprocessing.ipynb → notebooks/02_preprocessing.ipynb
# Create src/preprocessing.py (modular version)
git add .
git commit -m "feat(preprocess): add Winsorization, Yeo-Johnson transform, redundancy removal pipeline"
git push origin feat/preprocessing-pipeline
# → Create PR → merge

# Branch 3: Testing
git checkout main; git pull
git checkout -b feat/validation-tests
git add tests/
git commit -m "feat(tests): add pipeline output validation and data integrity checks"
git push origin feat/validation-tests
# → Create PR → merge
```

**Rajat's workflow:**
```powershell
# Branch 1: Feature Engineering
git checkout -b feat/feature-engineering
git add src/feature_engineering.py
git commit -m "feat(features): engineer 47 features — interactions, aggregates, polynomials, cross-group"
git push origin feat/feature-engineering
# → PR → merge

# Branch 2: Model
git checkout main; git pull
git checkout -b feat/model-training
git add src/model.py config.py
git commit -m "feat(model): add LightGBM/XGBoost/ExtraTrees with stratified 5-fold CV selection"
git push origin feat/model-training
# → PR → merge

# Branch 3: Pipeline + README
git checkout main; git pull
git checkout -b feat/pipeline-integration
git add run.py README.md
git commit -m "feat(pipeline): integrate end-to-end run.py with config-driven execution"
git push origin feat/pipeline-integration
# → PR → merge
```

---

## Commit Message Convention (Conventional Commits)

Judges notice professional commit messages:
```
feat(scope): description     ← new feature
fix(scope): description      ← bug fix
refactor(scope): description ← code restructuring
docs: description            ← documentation
chore: description           ← tooling/config
test: description            ← tests
```

---

## Key Tie-Breaker Wins

### 1. Code Optimization
- Feature engineering uses vectorized NumPy (already good)
- No unnecessary loops — pandas vectorized ops
- `n_jobs=-1` for parallel training (already done)
- Single `run.py` entry point — clean execution

### 2. Architectural Elegance
- **Modular design**: separate files for preprocessing, features, model
- **Config-driven**: all hyperparameters in `config.py`, not scattered
- **Type hints**: already present in `improved_model.py`
- **Docstrings**: well-documented functions

### 3. Overall Quality
- Professional README with results table, approach explanation
- `.gitignore`, `requirements.txt`, proper folder structure
- Tests for output validation
- Clean commit history with meaningful messages
- PR-based workflow showing collaboration

---

## Critical Things to Avoid

| Mistake | Why It Kills Your Score |
|---------|------------------------|
| Flat file structure (all files in root) | Judges see "no architecture" |
| Messy commit history (`fix`, `asdf`, `test123`) | Tie-breaker: "overall quality" |
| Single giant commit | Looks like one person did everything |
| No `.gitignore` (pushing `__pycache__`, `.ipynb_checkpoints`) | Unprofessional |
| No `requirements.txt` | Can't reproduce your work |
| Raw CSVs in git (large files) | Use `.gitignore` for raw data or Git LFS |

---

## Final Checklist: What Each Person Pushes

| Member | Files Owned | # Commits | # PRs |
|--------|------------|-----------|-------|
| **Rajat** | `config.py`, `run.py`, `src/feature_engineering.py`, `src/model.py`, `README.md` | ~5-6 | 3 |
| **Sidh** | `notebooks/01_eda.ipynb`, `notebooks/02_preprocessing.ipynb`, `src/preprocessing.py`, `tests/test_pipeline.py`, `requirements.txt`, `.gitignore` | ~5-6 | 3 |

> Both members have **roughly equal commits and PRs**, showing balanced team contribution. The modular architecture + clean git history will give you the edge on tie-breakers.

---

## TODO Tracker

### Rajat Pundir
- [ ] Create GitHub repo with folder structure
- [ ] Create `config.py` with all hyperparameters
- [ ] Create `src/feature_engineering.py` (from improved_model.py)
- [ ] Create `src/model.py` (from improved_model.py)
- [ ] Create `run.py` (single entry point)
- [ ] Write professional `README.md`
- [ ] Review & merge Sidh's PRs
- [ ] Final submission check

### Sidh Khurana
- [ ] Clone repo, accept collaborator invite
- [ ] Move `datacheck.ipynb` → `notebooks/01_eda.ipynb`
- [ ] Move `preprocessing.ipynb` → `notebooks/02_preprocessing.ipynb`
- [ ] Create `src/preprocessing.py` (modular version from notebook)
- [ ] Create `tests/test_pipeline.py`
- [ ] Create `requirements.txt` and `.gitignore`
- [ ] Create `notebooks/03_results_analysis.ipynb`
- [ ] Push all via feature branches with proper commit messages
