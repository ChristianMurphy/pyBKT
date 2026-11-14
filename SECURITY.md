# Security policy

## Supported versions

Security fixes go into the latest release on [PyPI](https://pypi.org/project/pyBKT/).

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub's vulnerability reporting form](https://github.com/CAHLR/pyBKT/security/advisories/new)
instead of a public issue. Include the pyBKT version, your Python version and
platform, and the steps to reproduce the problem.

## Verifying a release

Releases are built and published by the
[release workflow](.github/workflows/release.yml) through PyPI Trusted
Publishing. Each file on PyPI carries a signed attestation that ties it to this
repository and that workflow. To check a file:

```sh
pip install pypi-attestations
pypi-attestations verify pypi --repository https://github.com/CAHLR/pyBKT pypi:pybkt-1.4.3.tar.gz
```
