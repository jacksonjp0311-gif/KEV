# Security Policy

## Trust boundaries

- Never commit credentials, private keys, private transcripts, local state, or
  private reviewed lessons.
- The local HTTP surface binds to loopback, validates Host and Origin, accepts
  JSON writes only, limits request bodies, contains static paths, and returns
  sanitized internal errors. It is still a local research service, not a
  hardened multi-user or network deployment.
- Session identifiers are validated and contained beneath the session root.
- Registry/state-pinned checkpoint hashes are verified before deserialization
  by the current runtime. Evaluation hashes and deserializes one captured byte
  string, so its recorded model digest identifies the bytes actually loaded.
  Generic caller-supplied evaluation, training, and calibration paths still do
  not accept an externally expected digest; treat them as trusted local inputs
  and verify their hashes before use.
- Tool registries are empty by default. Inputs and outputs cross declared
  schemas, production mutation is rejected, every attempt gets a receipt, and
  one successful receipt cannot be replayed into multiple verified observations.
- Language, neural proposals, and generated narration have no execution or
  checkpoint-activation authority.

## Local evidence limits

State, sessions, and ledgers are plaintext local files. The implementation uses
atomic replacement, process/thread locking, and a pending state/event recovery
journal, but it does not configure OS ACLs or encrypt state. Run KEV under an
appropriately protected user account and state directory.

The hash-linked ledger and co-located head detect altered, reordered, or
removed events, a missing tail, and changes in ledger byte size. A same-size
formatting-only rewrite that leaves every parsed event unchanged is outside
that check. The records are not externally signed: a filesystem administrator
who can rewrite both the ledger and its anchor can forge local history. Use an
independently witnessed or signed anchor before relying on it across
adversarial trust boundaries.

`kev doctor` roots its shipped-artifact verification in the co-located public
model registry and follows every hash-pinned active-evidence link before trusting
the frozen manifest's contents. This detects an isolated artifact, manifest,
or evidence rewrite. The registry is not externally signed, so an attacker who
can coherently replace the registry and every downstream digest remains
outside this local integrity boundary.

Never describe a language response as proof of execution. Only a matching,
verifiable receipt and ledger event support that claim. Never overwrite an
incumbent checkpoint; activate only an exact hash that passed the documented
gate.

Report vulnerabilities through a private GitHub security advisory when
available. Do not publish secrets or private evidence in an issue.
