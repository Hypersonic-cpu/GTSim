"""Inspection and validation of canonical layered IR documents."""
import argparse
from pathlib import Path
from .serialization.json_codec import loads, digest
from .serialization.printer import print_ir


def main():
    parser=argparse.ArgumentParser(prog='gtsim-ir')
    parser.add_argument('command',choices=('inspect','verify'))
    parser.add_argument('--input',required=True,type=Path)
    args=parser.parse_args()
    try:
        obj=loads(args.input.read_text())
        print(print_ir(obj) if args.command=='inspect' else f'{obj.symbol}: valid {digest(obj)}')
    except (ValueError,TypeError,OSError) as error:
        parser.exit(2,f'{args.input}: {error}\n')


if __name__=='__main__':main()
