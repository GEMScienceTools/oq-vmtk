"""Sanity checks on packaging metadata and installation instructions.

These tests guard against regressions like the version string lagging the
release tag, or the README/installation docs referring to a requirements file
that no longer exists (the latter actually tripped a reviewer during
manuscript review).
"""

import os
import re
import unittest


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()


class TestPackagingMetadata(unittest.TestCase):

    def test_version_matches_citation(self):
        from openquake.vmtk import __version__

        cff = _read("CITATION.cff")
        m = re.search(r'^version:\s*"?([^"\s]+)"?', cff, flags=re.MULTILINE)
        self.assertIsNotNone(
            m, "CITATION.cff is missing a 'version:' field"
        )
        self.assertEqual(__version__, m.group(1),
                         "openquake.vmtk.__version__ disagrees with CITATION.cff")

    def test_referenced_requirements_files_exist(self):
        """Every requirements file mentioned in user-facing docs must exist."""
        sources = [
            _read("README.md"),
            _read("docsrc/contents/installation.rst"),
        ]
        pattern = re.compile(r"requirements-py\d+-[\w_]+\.txt")
        referenced = set()
        for text in sources:
            referenced.update(pattern.findall(text))

        self.assertTrue(referenced,
                        "no requirements-*.txt references found in docs")
        for name in sorted(referenced):
            full = os.path.join(REPO_ROOT, name)
            self.assertTrue(
                os.path.isfile(full),
                f"docs reference {name} but the file does not exist",
            )


if __name__ == "__main__":
    unittest.main()
