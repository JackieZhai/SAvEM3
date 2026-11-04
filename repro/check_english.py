"""Check publishable repository files for Han text and CJK punctuation.

Inspect tracked files and non-ignored new files, including hidden files and
Unicode escapes. Ignored local backups, environments, data, and Git history are
outside the publishing scope. This is a character check, not a language detector.
"""
import argparse
from pathlib import Path
import re
import subprocess
import unicodedata


ESCAPES = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|&#x([0-9a-fA-F]+);|&#([0-9]+);")


def forbidden_character(char):
    """Recognize Han characters and common CJK/fullwidth punctuation."""
    name = unicodedata.name(char, '')
    value = ord(char)
    return (name.startswith(('CJK UNIFIED IDEOGRAPH', 'CJK COMPATIBILITY IDEOGRAPH',
                             'CJK RADICAL', 'KANGXI RADICAL'))
            or 0x3000 <= value <= 0x303F
            or 0xFF00 <= value <= 0xFFEF)


def decode_escapes(text):
    """Decode Unicode escapes and numeric HTML entities without executing text."""
    def replace(match):
        groups = match.groups()
        code = next(group for group in groups if group is not None)
        try:
            return chr(int(code, 10 if groups[3] is not None else 16))
        except (ValueError, OverflowError):
            return match.group(0)

    decoded = ESCAPES.sub(replace, text)
    # Combine JSON UTF-16 surrogate pairs, including supplementary Han characters.
    return decoded.encode('utf-16-le', errors='surrogatepass').decode('utf-16-le', errors='surrogatepass')


def has_chinese_text(text):
    return any(forbidden_character(char) for char in decode_escapes(text))


def source_paths(root):
    output = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    return sorted({name.decode('utf-8') for name in output.split(b'\0') if name})


def audit(root):
    root = Path(root).resolve()
    failures = []
    checked = 0
    for name in source_paths(root):
        if has_chinese_text(name):
            failures.append(f'{name}: non-English filename')
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if data.startswith((b'\xff\xfe', b'\xfe\xff')):
            text = data.decode('utf-16')
        else:
            if b'\0' in data:
                continue  # Binary assets are not source text.
            try:
                text = data.decode('utf-8-sig')
            except UnicodeError as error:
                failures.append(f'{name}: cannot inspect text encoding ({error.reason})')
                continue
        checked += 1
        for number, line in enumerate(text.splitlines(), 1):
            if has_chinese_text(line):
                failures.append(f'{name}:{number}: Han text or CJK punctuation')
    return checked, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    checked, failures = audit(args.root)
    for failure in failures:
        print(failure)
    print(f'Checked {checked} publishable text files; found {len(failures)} language violations.')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
