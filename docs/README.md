# Project Timeline Documentation

This folder contains the 20-week project timeline for the Qwen3-VL optimization project.

## Files

- `project_timeline.tex` - Main LaTeX document with the complete 20-week timeline

## Compiling the LaTeX Document

### Prerequisites

You need a LaTeX distribution installed:
- **Windows:** [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)
- **Mac:** [MacTeX](https://www.tug.org/mactex/)
- **Linux:** `sudo apt-get install texlive-full` (or equivalent)

### Compilation

To compile the LaTeX document to PDF:

```bash
cd docs
pdflatex project_timeline.tex
pdflatex project_timeline.tex  # Run twice for proper references
```

Or use an online LaTeX compiler like [Overleaf](https://www.overleaf.com/).

### Output

The compilation will generate `project_timeline.pdf` in the `docs/` folder.

## Document Structure

The timeline is organized into 6 phases:

1. **Phase 1 (Weeks 1-4):** Setup and Baseline Establishment
2. **Phase 2 (Weeks 5-8):** Initial Optimization Techniques (Quantization, Pruning)
3. **Phase 3 (Weeks 9-12):** Advanced Optimization Techniques (Distillation, Architecture)
4. **Phase 4 (Weeks 13-16):** Runtime Optimizations
5. **Phase 5 (Weeks 17-18):** Comprehensive Evaluation
6. **Phase 6 (Weeks 19-20):** Documentation and Finalization

Each week includes detailed bullet points of tasks and deliverables.

