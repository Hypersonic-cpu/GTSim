#!/usr/bin/env python3
"""Repeat the two Bank/Plane outliers with the exact saved binary/config/graph."""
import argparse
import json
from pathlib import Path
import subprocess

from run_serve_hbf_backends import ROOT, parse, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build-backends')
    args = parser.parse_args()
    spec = json.loads((args.results/'run-spec.json').read_text())
    rows = json.loads((args.results/'results.json').read_text())
    binary = args.build_dir.resolve()/'gtsim_serve_ir'
    assert sha(binary) == spec['binary_sha256']
    config = ROOT/spec['configs']['hbfine']['path']
    assert sha(config) == spec['configs']['hbfine']['sha256']
    checks = []
    for name in ('gqa_qkvgate', 'mlp_gate_up'):
        previous = next(row for row in rows if row['name'] == name and row['backend'] == 'hbfine')
        graph = Path(spec['inputs'][name]['path'])
        assert sha(graph) == spec['inputs'][name]['sha256']
        folder = args.results/(name+'-hbfine')
        command = [str(binary), str(config), str(graph)]
        log = folder/'simulation-confirm.log'
        resource = folder/'resources-confirm.txt'
        if log.exists():
            raise RuntimeError('Refusing to overwrite confirmation '+str(log))
        with log.open('w') as handle:
            subprocess.run(['/usr/bin/time','-l','-o',str(resource),*command],stdout=handle,stderr=subprocess.STDOUT,check=True)
        repeated = parse(log)
        for field in ('cycles','drained_cycles','memory','hbfine','stages'):
            assert repeated[field] == previous[field], (name,field)
        checks.append(dict(name=name,cycles=repeated['cycles'],confirmed=True,log_sha256=sha(log)))
        write(args.results/'confirmations.json',checks)
        print(name,repeated['cycles'],'cycles, exact repeated timing/traffic/resource counters',flush=True)


if __name__ == '__main__':
    main()
