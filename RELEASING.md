# Releasing pyBKT

Publishing a GitHub release runs the
[release workflow](.github/workflows/release.yml), which builds the wheels and
the sdist and uploads them to PyPI. Nobody uploads from their own
machine, and there is no PyPI API token.

## Before the release

1. Check that the tests pass on `master`.
2. Set the new version in `setup.py`. It appears four times: `version` and
   `download_url`, in both `setup()` calls.
3. Merge that change to `master`.
4. Optional: run the release workflow by hand from the Actions tab
   ("Run workflow" on `master`). A manual run builds every file and
   generates the SBOM, but does not upload to PyPI.

## Release

1. Create a GitHub release from `master`. Name the tag after the version with
   no prefix, such as `1.5.0`, to match `download_url` and the earlier tags.
2. Publishing the release starts the workflow. If the `pypi` environment has
   required reviewers, a reviewer approves the "Publish to PyPI" job once the
   build jobs are green.

## What the workflow does

| Job | Result |
|---|---|
| Build wheels, Build sdist | Wheels for Linux and macOS from cibuildwheel, and the sdist |
| Generate SBOM, Attach SBOM | `pybkt-sbom.cdx.json`, a CycloneDX list of pyBKT and the dependency versions pip resolved, attached to the GitHub release |
| Publish to PyPI | Uploads through Trusted Publishing and adds a PyPI publish attestation to each file |

## After the release

1. On the project's [PyPI page](https://pypi.org/project/pyBKT/), check that
   the Security tab shows full provenance for the new version.
2. Verify one file from the command line:

   ```sh
   pip install pypi-attestations
   pypi-attestations verify pypi --repository https://github.com/CAHLR/pyBKT pypi:pybkt-1.5.0.tar.gz
   ```

3. Install the new version in a clean environment and import it.

## If the upload fails

PyPI never lets a file name be reused, so check what was uploaded before
trying again. If nothing was uploaded, fix the problem and re-run the failed
jobs. If some files were uploaded, don't re-run the workflow: fix the problem
and release the next patch version, so every file of a release comes from one
run.

## Access

- PyPI trusts one publisher: this repository, the workflow `release.yml` and
  the environment `pypi`. Renaming any of them breaks publishing until the
  publisher is updated in the project's PyPI settings.
- Keep required reviewers on the `pypi` environment, and two-factor
  authentication on every maintainer's GitHub and PyPI accounts.
- The workflows pin every action to a commit. Dependabot proposes updates, and
  zizmor audits the workflows on each pull request that changes them.
