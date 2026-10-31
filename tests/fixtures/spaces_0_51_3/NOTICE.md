# Upstream source used only for offline compatibility tests

Source: Hugging Face `spaces` 0.51.3, author Charles Bensimon.
Declared upstream license: Apache-2.0; see LICENSE.txt.
The wheel declares this license in METADATA but contains no separate LICENSE file;
the included license text comes from https://www.apache.org/licenses/LICENSE-2.0.txt.

The text fixtures contain exact function excerpts selected with Python AST from the
verified public PyPI wheel. Bodies are unmodified; other functions/imports are omitted.
`provenance.json` pins wheel, complete original-file and excerpt SHA-256 values.
Tests compile these functions with postponed annotations and stubbed package imports,
workers and transport, without importing spaces/__init__.py or CUDA emulation.
These fixtures are not installed, are not runtime dependencies, and are excluded
from the 17-file Space bundle. They do not validate the remote scheduler or hardware.
