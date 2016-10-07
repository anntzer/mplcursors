#!/bin/sh
set -e
# Cleanup.
if [[ -n $(git status -s) ]]; then
    echo 'There are uncommitted changes.'
    exit 1
fi
git clean -xdf
# Test.
py.test
# Test installation.
python setup.py bdist_wheel
VENVNAME="$(mktemp -u)"
trap 'rm -rf "$VENVNAME"' EXIT
python -mvenv $VENVNAME
(
    source $VENVNAME/bin/activate
    PIP_CONFIG_FILE=/dev/null pip --isolated install dist/*.whl
)
# Test docs.
(python setup.py build_ext -i &&
    cd doc &&
    make html)
# Ready to go?
if ! git describe --exact-match HEAD; then
    echo 'This commit is untagged.'
    exit 1
fi
# Yes.
echo 'Run `twine upload dist/*`'
