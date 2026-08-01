# Submission

## Memory mechanism

Describe the write rule and the read rule. Explain how one parameter set
supports all tested memory budgets.

Explain how the memory represents conditions, context scopes, actions, and
partial-policy priority within 24 bytes per slot.

Explain how the writer converts context-traffic estimates into residual rule
value after later policies remove parts of the scope.

## Public results

Checkpoint:

Sweep artifact:

Provide one matrix for each traffic and noise pair.

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |
| 64 | | | | | |

Each entry must use `overall accuracy [95% CI]`. Keep exact counts and subgroup
metrics in `artifacts/sweep.json`.

## Partial override behavior

Compare single-rule requests and conflict requests. Report evidence that the
writer preserves active parts of earlier policies.

Identify errors caused by rule compilation separately from retrieval errors.

## Allocation and subgroup behavior

Compare uniform and Zipf traffic. Compare exact and noisy estimates.

Report hot, cold, common-context, and rare-context accuracy. Explain which
information the writer discards when the payload is full.

## Prediction made before the holdout

- Predicted smallest `K` for 90% overall action accuracy:
- Functional family:
- Fit or calculation:
- Expected effect of 128 policies:
- Expected effect of the partial-policy-rate change:
- Expected effect of `crossing` scopes:
- Expected effect of log-normal traffic and `sigma=1.0` noise:
- Reason that the relationship can extrapolate:

Store the machine-readable commitment in `artifacts/prediction.json`.

## Holdout

Report overall, conflict, single-rule, hot, cold, common-context, and
rare-context curves.

Report the smallest tested `K` that reaches 90 percent. Report the absolute
prediction error when this value exists.

## Weaker approach or ablation

Describe one tested alternative. Identify its main limit.

Possible limits include compilation errors, lossy encoding, retrieval errors,
noisy allocation, optimization failure, and insufficient payload capacity.

## Objective and information allocation

Explain each training objective. State what the memory preserves and discards.
Connect these choices to measured subgroup results.

## Recommendation

Recommend a memory budget. State the supporting accuracy, uncertainty, and
known failure conditions.
