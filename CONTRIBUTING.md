# Contributing to KEV

KEV is experimental research software. Contributions must distinguish an
implemented mechanism, a measured result, and an aspiration.

Before changing code, read `AGENT_START_HERE.md`, `AI_AGENT_GUIDE.md`, and the
architecture, mathematics, roadmap, and model-evidence documents. Then run:

```text
python -m kev.cli doctor
python -m pytest -q -p no:cacheprovider tests_public
```

A model or evaluation pull request should state:

- the hypothesis and authority boundary;
- every changed file and schema;
- parent checkpoint, reviewed lesson, held-out vocabulary, calibration, and
  frozen-suite hashes;
- before/after fresh, retention, OOV, composition, and calibration results;
- raw failure and eval-card paths;
- whether the strict gate rejected or qualified the challenger; and
- limitations and any items that are no longer fresh evidence.

Never overwrite an incumbent, silently refresh an audit, treat chat as training
data, use training loss as promotion evidence, discard a rejected candidate, or
claim an unreceipted action occurred. New evaluation content requires a new
version. New training content requires explicit review and permission
provenance.

Keep large/private models, runtime state, transcripts, secrets, and private
lessons out of Git. Small public reference artifacts may be committed only with
documented hashes and provenance. A change is not complete until a clean clone
can verify its shipped artifacts and tests.
