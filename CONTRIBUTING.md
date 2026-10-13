# Contributing to vllm-rlt

Start with the [Developer Must-Read](docs/developer-guide.md) for the maintained
development workflow, validation requirements, and completion criteria.

Before implementation and again before PR submission, complete the human
[contributor-intent gate](.agents/skills/vllm-rlt-review-pr/SKILL.md#human-contributor-intent-gate)
in the repository's [author self-review skill](.agents/skills/vllm-rlt-review-pr/SKILL.md).
Reuse explicit, traceable human answers and ask only for missing decisions.
The gate applies to contributor work, not independent read-only reviews.

Keep the existing performance and code-quality requirements in the developer
guide, including the [performance skill](.agents/skills/rlt-perf-opt/SKILL.md)
and [refactoring skill](.agents/skills/rlt-refactor/SKILL.md). Scale evidence to
the changed behavior and disclose unrun checks; documentation-only changes do
not justify irrelevant GPU benchmarks.
