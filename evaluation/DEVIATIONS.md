# Deviations from the evaluation protocol (v1.1)

Protocol v1.1 was agreed with Dr. Wade on 21 November 2026 and frozen on 23 November 2026
(public record: release [`protocol-v1.1-frozen`](https://github.com/wandiyajames-arch/igcse_math_tutor/releases/tag/protocol-v1.1-frozen)).
Section 9 of the protocol requires every later change to be logged here with its date, reason and
expected effect, and listed in the paper's appendix.

| ID | Logged | Deviation | Reason | Expected effect |
|---|---|---|---|---|
| D1 | 26 Nov 2026 | The grader tag `gaie-eval-v1` was created on **26 November 2026 at 00:25 (UTC+1)**, not on 23 November as the protocol and the release say. It is an annotated tag in the private research repository, on commit `439ce321a9a5f06d9173f2c81201426d7a534de8` (tag object `6915324cdd5dcf51a804e9b9f860c53e55e2b837`). | The commit exists only in the research repository: its history was rewritten for the handover, so it could not be tagged in the older fork. Creating the tag took longer than planned. | **None on what is tested.** The commit is unchanged; only the tag is late. An independent copy of the repository received on 18 November 2026 (archive stamped with commit `439ce321…`) has the same `src/grading` tree `92ec6559913de06c7aeffd4130fa5487f3ac4ff8`, the same `src/models` tree `81a99f64ccc89f5fedecc6465368d1f39ca9b32f`, the same `requirements.txt` blob `f9c58993d3436082bfc64b27641972fd47d57c20` (SymPy 1.14.0), and the same harness grader fingerprint `2db5cf2dc1f506dd3c45975fe45005bba0e04be7e81a85e28896749b325d100d` (SHA-256 over the 15 grader source files) that the evaluation pipeline records in every run. |
| D2 | 26 Nov 2026 | The public freeze release moved from tag `protocol-v1.1-freeze` to `protocol-v1.1-frozen`. | On 23 November third-party copyrighted material (Cambridge papers) was removed from the public repository's entire history; the rewrite gave every commit a new ID, so the release was re-pointed to the cleaned commit with the same files. | None. The release text, its publication date and the protocol PDF fingerprint `b50852944f8fc2e88f6bc8510d97d70e2fb8ca6dab984ff7d5acd1e0c96d1206` are unchanged; the new tag's commit has the same files as the original one. |

## How to check D1

In the research repository:

```bash
git rev-parse 'gaie-eval-v1^{commit}'               # 439ce321a9a5f06d9173f2c81201426d7a534de8
git rev-parse 'gaie-eval-v1^{commit}:src/grading'   # 92ec6559913de06c7aeffd4130fa5487f3ac4ff8
git rev-parse 'gaie-eval-v1^{commit}:src/models'    # 81a99f64ccc89f5fedecc6465368d1f39ca9b32f
python -c "from evaluation.grader_adapter import grader_fingerprint; print(grader_fingerprint())"
# at that commit: {'flavour': 'angelah', 'files': 15, 'sha256': '2db5cf2d...5100d'}
```
