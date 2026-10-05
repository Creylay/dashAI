"""Stamp a test version and the fork's repository into the source tree.

Used only by the test-update-build workflow. The installers it builds report
TEST_VERSION and check the releases of the repository running the workflow
(the fork), so the in-app updater can be tried end to end.
"""

import os
import pathlib
import re

version = os.environ["TEST_VERSION"]
repository = os.environ["GITHUB_REPOSITORY"]

pyproject = pathlib.Path("pyproject.toml")
text, count = re.subn(
    r'(?m)^version = "[^"]+"', f'version = "{version}"', pyproject.read_text(), count=1
)
assert count == 1, "version line not found in pyproject.toml"
pyproject.write_text(text)

config = pathlib.Path("DashAI/back/config.py")
default = 'UPDATE_CHECK_REPOSITORY: str = "DashAISoftware/DashAI"'
text = config.read_text()
assert default in text, "UPDATE_CHECK_REPOSITORY default not found in config.py"
config.write_text(
    text.replace(default, f'UPDATE_CHECK_REPOSITORY: str = "{repository}"')
)

print(f"Building dashAI {version}, checking updates at {repository}")
