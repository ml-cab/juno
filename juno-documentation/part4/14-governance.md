(ch-14)=
# 14. Governance: Roles, Decision-Making, Code of Conduct

This chapter describes how the Juno project is governed: who holds which role, how decisions get
made, and how maintainers join or leave. The branching, feature, and release workflow that
maintainers execute is covered separately in [Chapter 15](#ch-15).

## Roles

**Maintainers** have write access to the repository, review and merge pull requests, and cut
releases. Current maintainers:

- Dmytro Soloviov (soulaway) — project lead
- Yevhen Soldatov (yevhensoldatov) — core maintainer

Maintainers make decisions by consensus. When consensus cannot be reached, the project lead has
a casting vote.

**Contributors** are anyone who has had a pull request merged, and are listed in
`CONTRIBUTORS.md`. Contributors do not have write access but are encouraged to review pull
requests and participate in design discussions.

**Users** are anyone using Juno. Users may open issues and participate in discussions.

## Decision-making

Routine decisions — bug fixes, minor features, dependency updates — are made by any maintainer
without requiring consensus.

Significant decisions — breaking API changes, new module additions, changes to license or
governance, or release of a new major version — require agreement from all active maintainers.
Proposals for significant changes are made via a GitHub issue labelled `proposal` and remain
open for at least seven days to allow community input.

## Adding and removing maintainers

A new maintainer may be nominated by an existing maintainer after:

- Sustained, high-quality contribution over at least three months.
- Familiarity with the codebase across at least two modules.
- Agreement from all existing maintainers.

A maintainer who is unresponsive for more than six months, or who requests to step down, is
moved to emeritus status. Emeritus maintainers are listed in `CONTRIBUTORS.md` with their status
noted, retain credit for their contributions, but no longer have write access.

## Releases

The full release and branching workflow — feature branch naming, merge review, publishing to
Maven Central, tagging, and preparing the next release branch — is documented in
[Chapter 15](#ch-15).

## Code of conduct

Contributors and maintainers are expected to engage respectfully. Harassment, discriminatory
language, and personal attacks are not tolerated in any project space: repository, Discord,
mailing list, or events.

Reports of conduct violations may be sent privately to [dev@ml.cab](mailto:dev@ml.cab).
Maintainers review reports promptly and respond within five business days.

## Amendments

Changes to this document require agreement from all active maintainers and are proposed via a
`proposal`-labelled GitHub issue with a minimum seven-day comment period, the same process used
for other significant decisions above.

---

[← Chapter 13: Performance Methodology](#ch-13) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 15: Contributing and the Release Process →](#ch-15)
