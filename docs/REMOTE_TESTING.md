# Remote Testing

The original request asks not to run locally. This repository therefore includes
GitHub Actions workflow templates that can validate the code remotely after the
project is pushed to GitHub.

The current `gh` token used to create the repository did not include the
`workflow` scope, so files under `.github/workflows/` could not be pushed
directly. The workflow files are therefore stored in `workflow-templates/`.

## CI

`workflow-templates/ci.yml` installs the package, runs Ruff, and runs Pytest.

## Sample Artifacts

`workflow-templates/samples.yml` generates draft-mode sample images and uploads
them as workflow artifacts. These prove the end-to-end interface and file output
path, not trained model quality.

## Enable Workflows

Copy the templates after refreshing GitHub auth with the `workflow` scope:

```bash
gh auth refresh -h github.com -s workflow
mkdir -p .github/workflows
cp workflow-templates/*.yml .github/workflows/
git add .github/workflows
git commit -m "Enable GitHub Actions workflows"
git push
```

## Manual Remote Commands

After pushing:

```bash
gh workflow run ci.yml
gh workflow run samples.yml
gh run list --limit 5
```

For trained checkpoints, use a GPU runner or external training platform and
publish:

- exact command;
- commit hash;
- checkpoint hash;
- generated samples;
- benchmark JSON;
- hardware and runtime.
