"""perfect_aliasing — the RL-deception train / probe / eval instrument.

The package is *import-closed*: modules flat-import their siblings (``import game``), so it runs
correctly when ``src/perfect_aliasing/`` is the working directory / on ``sys.path`` (local runs, Modal, and
single-folder uploads). See ``README.md``.
"""
