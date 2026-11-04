"""Private numeric worker protocol for optional SOFIMA registration environments."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from probe_em.sofima_registration import estimate, runtime_info
    if args.describe:
        print(json.dumps(runtime_info()))
        return
    if not all((args.input, args.parameters, args.output)):
        parser.error('Provide --input, --parameters and --output, or --describe')
    import numpy as np
    with np.load(args.input, allow_pickle=False) as data:
        field, report = estimate(data['raw'], data['resolution'], data['offset'],
                                 **json.loads(args.parameters.read_text()))
    field.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
