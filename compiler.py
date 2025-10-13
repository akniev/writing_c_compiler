#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path
from lexer import *
from typing import *
from parser import *
from assembly import *
from tacky import *
from asmgen import *
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="C compiler arguments parser")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lex", action="store_true", help="Run lexer stage only")
    group.add_argument("--parse", action="store_true", help="Run parser stage only")
    group.add_argument("--codegen", action="store_true", help="Run code generation stage only")
    group.add_argument("--tacky", action="store_true", help="Run TACKY generation stage only")

    parser.add_argument("path", type=str, help="Path to the source file")

    args = parser.parse_args()
    return args

def get_file_contents(file_path) -> str:
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main(argv):
    args = parse_args()
    text = get_file_contents(args.path)

    if args.lex:
        tokens = get_tokens(text)
        return 0
    
    if args.parse:
        tokens = get_tokens(text)
        ast = parse(tokens)

        ast.pretty_print()
        return 0
    
    if args.codegen:
        tokens = get_tokens(text)
        ast = parse(tokens)
        tacky_ast = t_parse_program(ast)
        asm_ast = tacky_parse_program(tacky_ast)
        asm_ast.pretty_print()
        return 0
    
    if args.tacky:
        tokens = get_tokens(text)
        ast = parse(tokens)
        tacky_ast = t_parse_program(ast)
        tacky_ast.pretty_print()
        return 0

    tokens = get_tokens(text)
    ast = parse(tokens)
    tacky_ast = t_parse_program(ast)
    asm_ast = tacky_parse_program(tacky_ast)
    asm_ast.pretty_print()
    asm = gen_asm(asm_ast)
    print(asm)

    asm_file = Path(args.path).with_suffix(".s")
    out_file = Path(args.path).with_suffix("")
    
    with open(asm_file, "w", encoding="utf-8") as f:
        f.write(asm)

    cmd = ["gcc", str(asm_file), "-o", str(out_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
    else:
        print("Compilation succeeded!")


if __name__ == "__main__":
    main(sys.argv)
