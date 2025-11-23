# 📑 Ablation Framework – Documentation Index

Complete guide to the Optuna + MLflow ablation study framework and supporting documentation.

## 🚀 Start Here

**New to the framework?** Follow this path:

1. **[QUICKSTART.md](evaluation/guards/ablation/QUICKSTART.md)** ← Start here (5 min)
   - Installation steps
   - 1-minute smoke test
   - Common commands

2. **[DELIVERY_SUMMARY.md](evaluation/guards/ablation/DELIVERY_SUMMARY.md)** (10 min)
   - What was built
   - Key capabilities
   - Quick reference

3. **[README.md](evaluation/guards/ablation/README.md)** (30 min)
   - Complete user guide
   - API reference
   - Configuration examples
   - Troubleshooting

4. **[example_ablation.py](evaluation/guards/ablation/example_ablation.py)** (read code)
   - Runnable examples
   - CLI argument reference
   - Best practices

---

## 📚 Documentation Map

### Framework Documentation (In `evaluation/guards/ablation/`)

| File | Time | Audience | Content |
|------|------|----------|---------|
| `QUICKSTART.md` | 5 min | First-time users | Get up and running |
| `README.md` | 30 min | End users | Complete guide + API |
| `DELIVERY_SUMMARY.md` | 10 min | Stakeholders | Summary of deliverables |
| `example_ablation.py` | - | All users | Executable examples |
| `requirements_ablation.txt` | - | Developers | Dependencies |

### Design & Architecture (In `improvement_docs/`)

| File | Time | Audience | Content |
|------|------|----------|---------|
| `ABLATION_STUDY_FRAMEWORKS_ANALYSIS.md` | 20 min | Decision makers | Why Optuna+MLflow chosen |
| `OPTUNA_MLFLOW_IMPLEMENTATION_SUMMARY.md` | 15 min | Project leads | What was delivered |
| `OPTUNA_MLFLOW_ABLATION_IMPLEMENTATION.md` | 30 min | Architects | Design decisions + technical depth |

---

## 🎯 Common Tasks

### "I want to run an ablation study"
→ Read: `QUICKSTART.md`, then run `example_ablation.py`

### "I need to understand the architecture"
→ Read: `OPTUNA_MLFLOW_ABLATION_IMPLEMENTATION.md`

### "I want to customize parameters or metrics"
→ Read: `README.md` → "Extensibility" section

### "I need to compare frameworks"
→ Read: `ABLATION_STUDY_FRAMEWORKS_ANALYSIS.md`

### "I want to integrate with MLflow"
→ Read: `README.md` → "Integration with MLflow"

### "I have implementation questions"
→ Read: `example_ablation.py` docstrings

---

## 📦 Code Files

### Core Framework Modules

```
evaluation/guards/ablation/
├── ablation_executor.py           # Main orchestrator (Optuna + MLflow)
├── fixtures_adapter.py            # Fixture caching & lifecycle
├── metrics_collector.py           # Metrics computation
├── results_analyzer.py            # Analysis & reporting
└── __init__.py                    # Public API
```

### Support Files

```
evaluation/guards/ablation/
├── example_ablation.py            # CLI entry point + examples
├── requirements_ablation.txt      # Dependencies
├── README.md                      # User guide
├── QUICKSTART.md                  # Quick start
└── DELIVERY_SUMMARY.md            # Deliverables overview
```

---

## 🔑 Key Classes & Functions

### Main Entry Point

```python
from evaluation.guards.ablation import AblationExecutor

executor = AblationExecutor(n_trials=36, n_seeds=20)
summary = executor.run(task_type="binary_classification", n_workers=8)
```

See: `ablation_executor.py` → `AblationExecutor` class

### Result Analysis

```python
from evaluation.guards.ablation import ResultsAnalyzer

analyzer = ResultsAnalyzer("./ablation_results")
sensitivity = analyzer.get_parameter_sensitivity()
best = analyzer.get_best_configurations(top_k=5)
analyzer.generate_report("report.txt")
```

See: `results_analyzer.py` → `ResultsAnalyzer` class

### Metrics Collection

```python
from evaluation.guards.ablation import MetricsCollector

coverage = MetricsCollector.compute_coverage_metrics(explainer, X_test, y_test, alpha)
quality = MetricsCollector.compute_explanation_quality_metrics(explainer, X_test, y_test)
```

See: `metrics_collector.py` → `MetricsCollector` class

### Fixtures Management

```python
from evaluation.guards.ablation import FixturesAdapter

adapter = FixturesAdapter(random_seed=42)
data = adapter.get_binary_classification_data()
model = adapter.get_binary_classifier()
explainer = adapter.create_guarded_explainer(task_type, guard_config)
```

See: `fixtures_adapter.py` → `FixturesAdapter` class

---

## 📈 Usage Examples

### Quick Test (6 trials)
```bash
python evaluation/guards/ablation/example_ablation.py \
  --task binary_classification \
  --n-trials 6 \
  --n-workers 1
```

See: `QUICKSTART.md` → "Quick Start (5 minutes)"

### Full Ablation (36 trials)
```bash
python evaluation/guards/ablation/example_ablation.py \
  --task binary_classification \
  --n-trials 36 \
  --n-workers 1 \
  --generate-report \
  --export-csv
```

See: `README.md` → "Configuration Examples"

### With All Seeds (720 trials)
```bash
python evaluation/guards/ablation/example_ablation.py \
  --task binary_classification \
  --n-trials 720 \
  --n-seeds 20 \
  --n-workers 8 \
  --timeout 3600 \
  --generate-report \
  --export-csv
```

See: `README.md` → "Advanced Usage"

---

## 🎓 Learning Path

### Level 1: Quick Start (15 min)
1. Read `QUICKSTART.md`
2. Run `python example_ablation.py --n-trials 6`
3. View `ablation_results/ablation_summary.json`

### Level 2: User Proficiency (1 hour)
1. Read `README.md`
2. Run full ablation `--n-trials 36`
3. Use `ResultsAnalyzer` to analyze results
4. Generate report `--generate-report`

### Level 3: Extension & Customization (2-3 hours)
1. Read `OPTUNA_MLFLOW_ABLATION_IMPLEMENTATION.md`
2. Review source code in `ablation_executor.py`, etc.
3. Add custom metrics or modify parameters
4. Run with your changes

### Level 4: Architecture & Design (4-6 hours)
1. Read `OPTUNA_MLFLOW_ABLATION_IMPLEMENTATION.md` in depth
2. Study `ablation_executor.py` design patterns
3. Review `FixturesAdapter` caching strategy
4. Understand `ResultsAnalyzer` analysis pipeline

---

## 🔄 Workflow

### Typical User Workflow

```
1. Install dependencies
   ↓
2. Run smoke test (6 trials)
   ↓
3. View results in MLflow UI / JSON files
   ↓
4. Run full ablation (36 trials)
   ↓
5. Analyze results:
   - Sensitivity analysis
   - Best configurations
   - Export to CSV
   ↓
6. Generate report
   ↓
7. (Optional) Run with seeds (720 trials) for statistical rigor
```

See: `QUICKSTART.md` for step-by-step commands

### Developer Workflow

```
1. Understand architecture (read IMPLEMENTATION doc)
   ↓
2. Run smoke test to verify setup
   ↓
3. Extend metrics / modify parameters
   ↓
4. Test changes with small ablation (6-12 trials)
   ↓
5. Run full validation
   ↓
6. Document changes
```

---

## 📊 File Locations Quick Reference

| What | Where |
|------|-------|
| Framework code | `evaluation/guards/ablation/*.py` |
| Quick start | `evaluation/guards/ablation/QUICKSTART.md` |
| Full docs | `evaluation/guards/ablation/README.md` |
| Examples | `evaluation/guards/ablation/example_ablation.py` |
| Dependencies | `evaluation/guards/ablation/requirements_ablation.txt` |
| Design docs | `improvement_docs/OPTUNA_MLFLOW*.md` |
| Framework comparison | `improvement_docs/ABLATION_STUDY_FRAMEWORKS_ANALYSIS.md` |
| Results (after run) | `./ablation_results/` |

---

## ❓ Frequently Asked Questions

### Q: Where do I start?
**A:** Read `QUICKSTART.md` (5 min), then run the example.

### Q: How do I run an ablation?
**A:** `python example_ablation.py --task binary_classification --n-trials 36`

### Q: Where are results stored?
**A:** `ablation_results/` directory + MLflow dashboard (if running)

### Q: Can I customize parameters?
**A:** Yes! See `README.md` → "Extensibility" or edit `ablation_executor.py`

### Q: How long does it take?
**A:** 6 trials: 2 min | 36 trials: 15-30 min | 720 trials: 1-2 hours (with parallelization)

### Q: Can I run in parallel?
**A:** Yes! Use `--n-workers 8` (or your CPU count)

### Q: Can I resume interrupted runs?
**A:** Yes! Optuna persists state; just rerun the command

### Q: Do I need MLflow?
**A:** No, but recommended. Results are still saved to JSON files.

See: `README.md` → "Troubleshooting" for more Q&A

---

## 🎁 Deliverables Summary

**Code**: ~1,145 lines of production-ready Python  
**Documentation**: ~800 lines of comprehensive markdown  
**Examples**: Fully runnable CLI with 4+ usage patterns  
**Integration**: MLflow, JSON export, CSV export, report generation  
**Coverage**: Binary/multiclass classification + regression  
**Extensibility**: Custom metrics, custom samplers, Ray Tune ready  

**Status**: ✅ Ready for production use

---

## 🚀 Next Steps

1. **Install**: `pip install -r evaluation/guards/ablation/requirements_ablation.txt`
2. **Read**: `evaluation/guards/ablation/QUICKSTART.md`
3. **Run**: `python evaluation/guards/ablation/example_ablation.py --n-trials 6`
4. **Explore**: Check `ablation_results/ablation_summary.json`
5. **Analyze**: Use `ResultsAnalyzer` for deeper insights

---

## 📞 Support & Help

- **Getting started?** → `QUICKSTART.md`
- **Need help?** → `README.md` → "Troubleshooting"
- **API questions?** → `README.md` → "API Reference"
- **Design questions?** → `OPTUNA_MLFLOW_ABLATION_IMPLEMENTATION.md`
- **CLI options?** → `example_ablation.py --help`

---

**Version**: 1.0  
**Date**: November 23, 2025  
**Status**: Production Ready ✅
