"""Read-only source inventory across the multi-repository research workspace."""
import argparse
import ast
import json
import os
from pathlib import Path

SKIP = {'node_modules', '__pycache__', 'site-packages', 'checkpoints', 'pretrained_checkpoint',
        'prepared_embedding', 'prepared_segments_mul_mem', 'prepared_segments_mul_2d',
        'outputs', 'work_dirs', 'trained_model', 'volumes_local', 'volumes_local_aligned'}
EXTENSIONS = {'.py', '.pyi', '.ipynb', '.sh', '.md', '.rst', '.yaml', '.yml', '.json',
              '.toml', '.cfg', '.ini', '.txt', '.tex', '.c', '.cpp', '.h', '.hpp', '.cu',
              '.js', '.jsx', '.ts', '.tsx', '.html', '.css'}


def inventory(root):
    root = Path(root).resolve()
    modules, errors = {}, []
    for parent, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if not name.startswith('.')
                                and not name.startswith(('trace_results', 'demo_session')) and name not in SKIP)
        if 'info' in names:  # Neuroglancer chunks are data, not source files.
            directories[:] = []
            continue
        for name in sorted(names):
            path = Path(parent) / name
            if path.suffix.lower() not in EXTENSIONS or path.is_symlink():
                continue
            relative = path.relative_to(root)
            module = relative.parts[0] if len(relative.parts) > 1 else '(workspace)'
            entry = modules.setdefault(module, {'files': 0, 'python_files': 0, 'lines': 0, 'readmes': 0})
            try:
                source = path.read_text(encoding='utf-8-sig')
            except (UnicodeError, OSError):
                continue
            entry['files'] += 1
            entry['lines'] += len(source.splitlines())
            entry['readmes'] += int(name.lower().startswith('readme'))
            if path.suffix == '.py':
                entry['python_files'] += 1
                try:
                    ast.parse(source, filename=str(relative))
                except SyntaxError as error:
                    errors.append({'file': str(relative), 'line': error.lineno, 'error': error.msg})
    return {'root': str(root), 'scope': 'source/docs; excludes environments, cached volumes, weights and archives',
            'modules': modules, 'syntax_errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output')
    args = parser.parse_args()
    result = inventory(args.root)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
