"""Check the released aggregates and render figures, without loading models.

Run with the dependencies in requirements-analysis.txt. Outputs are published only
if the checksum, assertions, tests and figure generation all succeed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('experiments/002-generalization-dynamics')
DATA = EXPERIMENT / 'data/e3_consolidated.json'
EXPECTED_SHA256 = '349de0200b5a8e9a9649e79e82954f17ffd490f02a3768d525a69e68143d2c57'
SEEDS = ['cbid_gemma-9b_em', 'cbid_gemma-9b_em_s1', 'cbid_gemma-9b_em_s2']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(blob):
    cells = blob['identification']
    unique = {}
    for cell in cells.values():
        rows = [r for r in cell.get('curve', [])
                if r.get('action_ally') is not None and r.get('truth_ally') is not None]
        if rows:
            unique.setdefault(json.dumps(rows, sort_keys=True), rows)
    pairs = [r for rows in unique.values() for r in rows]
    worst = max(abs(r['action_ally'] - (1 - r['truth_ally'])) for r in pairs)
    if len(unique) != 39 or len(pairs) != 751 or worst > 1e-12:
        raise ValueError('The saved identity count or tolerance differs from the paper.')
    def final(cell, metric):
        return max((r for r in cell['curve'] if r.get(metric) is not None),
                   key=lambda r: r['l'])[metric]
    ally = [final(cells[k], 'truth_ally') for k in SEEDS]
    mixed = [final(cells[k], 'truth_mixed') for k in SEEDS]
    return {
        'identity': {'distinct_full_curve_records': len(unique), 'cell_layer_pairs': len(pairs),
                     'maximum_absolute_deviation': worst},
        'headline': {'cells': SEEDS, 'ally_fit_auroc': ally,
                     'ally_fit_mean': statistics.mean(ally),
                     'ally_fit_sample_sd': statistics.stdev(ally),
                     'mixed_fit_auroc': mixed,
                     'rival_deception_rates': [cells[k]['behavior']['rival_deception_rate'] for k in SEEDS]},
        'scope': 'Verification of saved aggregate results; no model rerun or external generalization test.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduce/output',
                        help='New output directory; existing paths are never overwritten.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose a new --output directory.')
    if sha256(ROOT / DATA) != EXPECTED_SHA256:
        raise ValueError('Dataset SHA256 mismatch; refusing to validate a different results file.')
    report = summarize(json.loads((ROOT / DATA).read_text()))
    report.update(dataset_sha256=EXPECTED_SHA256, python=platform.python_version(),
                  packages={name: importlib.metadata.version(name)
                            for name in ['numpy', 'matplotlib', 'pytest']})
    with tempfile.TemporaryDirectory(prefix='perfect-aliasing-verify-') as tmp:
        scratch = Path(tmp)
        artifact = scratch / 'artifact'
        for rel in [DATA, EXPERIMENT / 'make_figures.py', EXPERIMENT / 'figure_layout.py']:
            dest = artifact / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        shutil.copytree(ROOT / 'tests', artifact / 'tests',
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        env = os.environ.copy()
        env.update(PAPER='1', MPLCONFIGDIR=str(scratch / 'mpl'),
                   PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        results = scratch / 'results'
        results.mkdir()
        for name, command in [
            ('tests', [sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider']),
            ('render', [sys.executable, str(EXPERIMENT / 'make_figures.py')]),
        ]:
            run = subprocess.run(command, cwd=artifact, env=env, capture_output=True, text=True)
            log = run.stdout + run.stderr
            (results / f'{name}.txt').write_text(log)
            if run.returncode:
                print(log, file=sys.stderr)
                raise RuntimeError(f'{name} failed (exit {run.returncode}); no success report written.')
        figures = artifact / 'paper/figures'
        expected = {p.name for p in (ROOT / 'paper/figures').glob('*.png')}
        actual = {p.name for p in figures.glob('*.png')}
        if len(expected) != 15 or actual != expected:
            raise ValueError('Regenerated figure set differs from the 15 released figures.')
        report['figures'] = {name: {'sha256': sha256(figures / name),
                                   'matches_released_bytes': sha256(figures / name) == sha256(ROOT / 'paper/figures' / name)}
                             for name in sorted(actual)}
        report['figure_comparison_note'] = 'Byte differences can reflect fonts/platforms; inspect differences before claiming identical figures.'
        report['status'] = 'PASS: checksum, aggregate assertions, tests and figure generation'
        shutil.copytree(figures, results / 'figures')
        (results / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(results, output)
    print(f"PASS: {report['identity']['cell_layer_pairs']} cell-layer pairs; "
          f"max deviation {report['identity']['maximum_absolute_deviation']:.2e}")
    print(f"Headline ally-fit AUROC {report['headline']['ally_fit_mean']:.3f} +/- "
          f"{report['headline']['ally_fit_sample_sd']:.3f}; mixed-fit {report['headline']['mixed_fit_auroc']}")
    matches = sum(v['matches_released_bytes'] for v in report['figures'].values())
    print(f'15 figures regenerated; {matches}/15 byte-identical to release. Report: {output / "summary.json"}')


if __name__ == '__main__':
    main()
