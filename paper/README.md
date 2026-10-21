# Paper: reproducing tables and figures

Every number in `paper/manuscript/main.tex` and `paper/results/` is computed from **saved outputs only**
(no GPU, no new generation). Data is loaded through `paper/scripts/phase14_data.py`; see
`phase14/DATA_CATALOG.md` for where each file lives and `phase14/DATA_MANIFEST.json` for SHA-256 hashes.
Hidden states are on `F:\AGI_data\phase14_hidden` (junction `outputs/phase14_hidden`).

Environment: analysis scripts use `.venv-verify310` (Python 3.10, numpy, scikit-learn, sympy); the render
script needs only matplotlib. Run everything from the repo root. Rules for every exploratory analysis are in
`phase14/RUN_STATUS.md` (amendment 2026-10-15) and were written before the analysis was run.

```sh
PY=.venv-verify310/Scripts/python.exe
export PYTHONIOENCODING=utf-8

# Preregistered endpoints and the 16-cell grid (Table 1, Table 2): official report from the GPU run
#   outputs/phase14_remote/outputs/phase14_v1/main/analysis/{report.json,probe_report.json,report_tables.md}
$PY paper/scripts/phase14_preliminary.py      # local recomputation of all cells -> results/phase14_preliminary.{json,md}
$PY paper/scripts/phase14_decomposition.py    # 3-factor Shapley (bias / cut / invented) -> results/phase14_decomposition.*
$PY paper/scripts/phase14_truncation.py       # labels of capped outputs -> results/phase14_truncation.*

# Section 4.2, 4.5, Appendix A (~2 h CPU, bootstrap)
$PY paper/scripts/phase14_paper_tables.py     # -> results/phase14_paper_tables.json, results/rescued_audit_sample.csv
python paper/scripts/phase14_render_tables.py # -> results/phase14_{invalid_types,decomp4,practical,counts}.md,
                                              #    figures/invalid_types_*.{png,pdf}, figures/practical_calls.*,
                                              #    manuscript/tables/{decomp4,practical,counts}.tex

# Probe-weighted voting out of sample (Section 4.4; ~1 h CPU, reads hidden states)
$PY paper/scripts/phase14_f2_oos.py           # -> results/phase14_f2_oos.{json,md}

# Hidden-state probes: the paper uses the official probe_report.json from the GPU run. A local CPU re-run
# is possible but has not been done (no results/phase14_probe_local.json yet):
# $PY paper/scripts/phase14_probe_local.py    # -> results/phase14_probe_local.json
```

| Paper element | Source file |
|---|---|
| Table 1 (grid), Table 2 (Ren et al. comparison) | `report.json`, `results/phase14_preliminary.md` |
| Figure 1, seven outcome classes | `results/phase14_invalid_types.md`, `figures/invalid_types_base.pdf` |
| Table 3, four-factor split; sensitivity paragraph | `results/phase14_decomp4.md` |
| Table 4, Figure 2, practical value | `results/phase14_practical.md`, `figures/practical_calls.pdf` |
| Hidden states, F2 out of sample | `probe_report.json`, `results/phase14_f2_oos.md` |
| Appendix A counts, rescued check | `results/phase14_counts.md`, `results/phase14_rescued_audit.md` |
| Appendix B–C (Self_Correction, bridge) | `CHECKLIST_RESULTS.md`, `summary/SUMMARY_ALL.md` |
| Appendix D (truncation, 1536 tokens) | `results/phase14_truncation.md`, `phase14/RUN_STATUS.md` |

Compile the manuscript from `paper/manuscript/` with `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(no LaTeX installation was available on the analysis machine, so the source has not been compiled yet).

## arXiv package

`paper/arxiv/` is a flat copy of the manuscript for submission (figure paths changed from `../figures/` to `figures/`;
only `main.tex`, `refs.bib`, `tables/*.tex` and the two PDF figures) and `paper/arxiv_submission.zip` is the same folder zipped.
Regenerate the tables and figures first (commands above), then copy; fill the `\authorinfo{}` fields in `main.tex` before submitting.
