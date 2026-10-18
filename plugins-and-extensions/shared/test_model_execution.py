#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for model_execution.py and its sync with lib/model_execution.lua."""

import re
import sys
import unittest
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_execution as me  # noqa: E402

LUA_PATH = Path(__file__).resolve().parent / "lib" / "model_execution.lua"


def _lua_table_keys(lua_text: str, table_name: str) -> set[str]:
    """Extract top-level model keys of M.<table_name> = { ... } from the Lua
    source via the same brace-depth scan used to bound the block, then pull
    every `key = {` assignment at depth 1 inside it."""
    start = lua_text.index(f"M.{table_name} = {{")
    depth = 0
    i = lua_text.index("{", start)
    j = i
    while True:
        if lua_text[j] == "{":
            depth += 1
        elif lua_text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    block = lua_text[i + 1:j]
    # Only keys immediately followed by " = {" at nesting depth 0 within block
    keys = set()
    depth = 0
    k = 0
    pattern = re.compile(r"(\w+)\s*=\s*\{")
    for m in pattern.finditer(block):
        # depth of this match: count braces before it
        prefix = block[:m.start()]
        d = prefix.count("{") - prefix.count("}")
        if d == 0:
            keys.add(m.group(1))
    return keys


class TestRegistryBasics(TestCase):

    def test_is_local_supported_known_models(self):
        self.assertEqual(me.is_local_supported("midigen", "text2midi"), (True, me.MIDIGEN["text2midi"]["reason"]))
        self.assertEqual(me.is_local_supported("midigen", "midi_llm")[0], False)
        self.assertEqual(me.is_local_supported("text2audio", "sao")[0], True)
        self.assertEqual(me.is_local_supported("text2audio", "mustango")[0], False)
        self.assertEqual(me.is_local_supported("stemseparator", "sam_base")[0], True)
        self.assertEqual(me.is_local_supported("stemseparator", "htdemucs")[0], True)

    def test_unknown_domain_is_unsupported_not_raising(self):
        supported, reason = me.is_local_supported("not_a_domain", "sao")
        self.assertFalse(supported)
        self.assertIn("Unknown domain", reason)

    def test_unknown_model_is_unsupported_not_raising(self):
        supported, reason = me.is_local_supported("midigen", "not_a_model")
        self.assertFalse(supported)
        self.assertIn("Unknown model", reason)

    def test_every_entry_has_bool_and_nonempty_reason(self):
        for domain_name, table in (("midigen", me.MIDIGEN), ("text2audio", me.TEXT2AUDIO),
                                    ("stemseparator", me.STEMSEPARATOR)):
            for key, entry in table.items():
                self.assertIsInstance(entry["local_supported"], bool,
                                      f"{domain_name}.{key}.local_supported must be bool")
                self.assertTrue(entry["reason"].strip(),
                                f"{domain_name}.{key}.reason must be non-empty")


class TestLuaPythonSync(TestCase):
    """Guards against the Lua and Python registries drifting apart — both
    files are maintained by hand (no shared codegen), so this is the only
    thing that would catch a key added/removed on one side but not the other."""

    def setUp(self):
        self.lua_text = LUA_PATH.read_text()

    def test_midigen_keys_match(self):
        lua_keys = _lua_table_keys(self.lua_text, "midigen")
        self.assertEqual(lua_keys, set(me.MIDIGEN.keys()))

    def test_text2audio_keys_match(self):
        lua_keys = _lua_table_keys(self.lua_text, "text2audio")
        self.assertEqual(lua_keys, set(me.TEXT2AUDIO.keys()))

    def test_stemseparator_keys_match(self):
        lua_keys = _lua_table_keys(self.lua_text, "stemseparator")
        self.assertEqual(lua_keys, set(me.STEMSEPARATOR.keys()))

    def test_local_supported_values_match(self):
        for table_name, py_table in (("midigen", me.MIDIGEN), ("text2audio", me.TEXT2AUDIO),
                                      ("stemseparator", me.STEMSEPARATOR)):
            start = self.lua_text.index(f"M.{table_name} = {{")
            end = self.lua_text.index("\n}\n", start)
            block = self.lua_text[start:end]
            for key, entry in py_table.items():
                # Find this model's sub-block and check its local_supported literal
                key_start = block.index(f"{key} = {{", block.index(f"\n  {key} = {{") - 1)
                sub = block[key_start:key_start + 400]
                expected = "true" if entry["local_supported"] else "false"
                self.assertIn(
                    f"local_supported = {expected}", sub,
                    f"{table_name}.{key}: Lua local_supported should be {expected}",
                )


if __name__ == "__main__":
    unittest.main()
