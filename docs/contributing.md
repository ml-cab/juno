# Contributing to Juno

This document covers the branching model, feature contribution workflow, and release process.

For roles, decision-making authority, and governance, see [GOVERNANCE.md](GOVERNANCE.md).

---

## Branching model

| Branch | Purpose |
|--------|---------|
| `main` | Stable, released code only. Never committed to directly. |
| `release-X.Y.Z` | Ongoing development for an upcoming release. All features target this branch. |
| `NNN-short-description` | Short-lived feature branch. Branched from and merged back into the active release branch. |

---

## Contributing a feature

### 1. Sync the active release branch

```bash
git checkout release-0.1.1
git fetch
git pull origin release-0.1.1
```

### 2. Create a feature branch

Branch names follow the pattern `NNN-short-description`, where `NNN` is the issue number from the [project tracker](https://github.com/orgs/ml-cab/projects/1/views/1).

```bash
git checkout -b 43-fix-windows-startup release-0.1.1
```

### 3. Implement and push

```bash
git add scripts/run.bat
git commit -m "#43 fix dots in offending blocks on Windows"
git push origin 43-fix-windows-startup
```

### 4. Open a merge request

Open an MR from `43-fix-windows-startup` into the active release branch (e.g. `release-0.1.1`) via the [branch comparison page](https://github.com/ml-cab/juno/compare) — click the green `Create pull request` button after selecting branches.

- The feature must be reviewed by at least one maintainer.
- The feature branch is squash-merged (preserving the issue number) and then deleted.

---

## Release process

Releases are cut from the release branch into `main` by a maintainer.

### Step 1. Merge the release branch into `main`

Open an MR from `release-0.1.0` into `main` via the [branch comparison page](https://github.com/ml-cab/juno/compare). After review, merge and run a smoke test.

### Step 2. Publish to Maven Central

```bash
mvn clean verify
mvn clean deploy -Prelease-sign,central-publish -DskipTests
```

### Step 3. Tag the release

```bash
git tag -a v0.1.0 -m "Juno-0.1.0"
git push origin --tags
```

### Step 4. Create a GitHub Release

Create a new Release in the Git UI, attaching the version tag and a description of changes. Release notes live in [RELEASE_NOTES.md](../RELEASE_NOTES.md).

### Step 5. Prepare the next release branch

```bash
git checkout -b release-0.1.1 main
git push origin release-0.1.1
```

All subsequent feature branches target `release-0.1.1` from this point forward.