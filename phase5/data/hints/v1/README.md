# Rejected hint parser V1

This generated audit is not an experiment input. Train/development inspection
found false positives from chained equalities, percentages, approximate decimal
equalities, and algebraic expressions. No protected row was manually inspected.

The frozen replacement is V2, built by
`phase5/scripts/audit_hint_feasibility.py`. V1 is retained only to document why
the stricter rule was introduced before review generation.
