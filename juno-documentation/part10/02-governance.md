(ch-10-2)=
# 10.2. Governance

This document describes how the Juno project is governed, how decisions are made,
and how new maintainers join or leave the project.

---

## Roles

### Maintainers

Maintainers have write access to the repository, review and merge pull requests,
and cut releases. Current maintainers:

- Dmytro Soloviov (soulaway): project lead
- Yevhen Soldatov (yevhensoldatov): core maintainer

Maintainers make decisions by consensus. When consensus cannot be reached, the
project lead has a casting vote.

### Contributors

Anyone who has had a pull request merged is a contributor. Contributors are listed
in [Contributors](#ch-10-6). Contributors do not have write access but
are encouraged to review pull requests and participate in design discussions.

### Users

Anyone using Juno. Users may open issues and participate in discussions.

---

## Decision-Making

Routine decisions (bug fixes, minor features, dependency updates) are made by any
maintainer without requiring consensus.

Significant decisions (breaking API changes, new module additions, changes to
license or governance, release of a new major version) require agreement from all
active maintainers. Proposals for significant changes are made via a GitHub issue
labelled `proposal` and remain open for at least seven days to allow community input.

---

## Adding and Removing Maintainers

A new maintainer may be nominated by an existing maintainer after:

- Sustained, high-quality contribution over at least three months.
- Familiarity with the codebase across at least two modules.
- Agreement from all existing maintainers.

A maintainer who is unresponsive for more than six months, or who requests to step
down, is moved to emeritus status. Emeritus maintainers are listed in
the contributors list with their status noted. Emeritus maintainers retain credit for
their contributions but no longer have write access.

---

## Releases
 
The full release and branching workflow is documented in [contributing.md](#ch-10-1).
 
---

## Code of Conduct

Contributors and maintainers are expected to engage respectfully. Harassment,
discriminatory language, and personal attacks are not tolerated in any project
space (repository, Discord, mailing list, or events).

Reports of conduct violations may be sent privately to dev@ml.cab. Maintainers
will review reports promptly and respond within five business days.

---

## Amendments

Changes to this document require agreement from all active maintainers and are
proposed via a `proposal`-labelled GitHub issue with a minimum seven-day comment
period.

---

[<- 10.1 Contributing](#ch-10-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [10.3 Security Policy ->](#ch-10-3)
