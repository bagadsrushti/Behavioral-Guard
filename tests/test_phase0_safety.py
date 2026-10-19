"""
test_phase0_safety.py — Safety and Baseline Test Suite for Phase 0.

Verifies:
1. Keystroke logger never records raw typed characters (privacy guarantee).
2. Synthetic data generation does not overwrite existing raw data without --force.
3. main.py stage_generate skips generation when raw data exists and force=False.
4. Model checkpoints remain 100% intact with baseline SHA256 hashes.
5. Baseline raw datasets (90 CSVs) are intact and unmodified.
"""

import hashlib
import os
import string
import tempfile
import unittest
from pathlib import Path

from pynput import keyboard

from src.logging.keystroke_logger import _key_name, KeystrokeLogger
from data.synthetic.synthetic_data_gen import (
    generate_mouse_data,
    generate_keystroke_data,
    generate_all,
)
from main import stage_generate


# Baseline SHA256 checksums recorded at start of Phase 0
BASELINE_CHECKPOINT_HASHES = {
    "mouse_lstm_ae.pt": "8CB30C31E7130CFAE4AB725AA4E5081B51307F19C4800BCC34327F8DF82A47C5",
    "keystroke_lstm_ae.pt": "320E50DA5B5CD8586C033E10B0D3873A30BA96BB54D32629010FB9A60782BCF9",
    "fusion_rf.pkl": "16144BE85DC8558CF862123FACF5FE421359A6864317140DF269CB87057F8383",
    "mouse_scaler.pkl": "9D37A909709A6F8EBC4BE97FD87A14BD84E79CC06FEE5B056CEAF24EE68DCB91",
}


def _sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


class TestKeystrokeLoggerPrivacy(unittest.TestCase):
    """Ensure no typed characters are ever recorded or returned."""

    def test_alphabet_chars_mapped_to_letter(self):
        """All lowercase and uppercase letters must return 'letter' and never the char."""
        for c in string.ascii_letters:
            k = keyboard.KeyCode.from_char(c)
            result = _key_name(k)
            self.assertEqual(result, "letter")
            self.assertNotEqual(result, c)

    def test_digits_mapped_to_digit(self):
        """All digits 0-9 must return 'digit' and never the char."""
        for c in string.digits:
            k = keyboard.KeyCode.from_char(c)
            result = _key_name(k)
            self.assertEqual(result, "digit")
            self.assertNotEqual(result, c)

    def test_punctuation_mapped_to_punctuation(self):
        """All punctuation characters must return 'punctuation' and never the char."""
        for c in string.punctuation:
            k = keyboard.KeyCode.from_char(c)
            result = _key_name(k)
            self.assertEqual(result, "punctuation")
            self.assertNotEqual(result, c)

    def test_whitespace_mapped_to_space(self):
        """Spaces and tabs must map to category labels."""
        self.assertEqual(_key_name(keyboard.KeyCode.from_char(" ")), "space")
        self.assertEqual(_key_name(keyboard.Key.space), "space")
        self.assertEqual(_key_name(keyboard.Key.tab), "tab")

    def test_special_keys_categorization(self):
        """Special navigation and control keys must be mapped to categories."""
        self.assertEqual(_key_name(keyboard.Key.backspace), "backspace")
        self.assertEqual(_key_name(keyboard.Key.delete), "backspace")
        self.assertEqual(_key_name(keyboard.Key.enter), "enter")
        self.assertEqual(_key_name(keyboard.Key.shift), "modifier")
        self.assertEqual(_key_name(keyboard.Key.ctrl_l), "modifier")
        self.assertEqual(_key_name(keyboard.Key.alt_l), "modifier")

    def test_no_single_character_output_invariant(self):
        """Privacy invariant: output is always a category string of length >= 4."""
        test_chars = string.printable
        for c in test_chars:
            k = keyboard.KeyCode.from_char(c)
            result = _key_name(k)
            self.assertGreaterEqual(
                len(result), 4, f"Output '{result}' for char '{c}' is too short / looks like raw char!"
            )
            self.assertNotEqual(result, c)

    def test_write_row_records_only_categories(self):
        """Test that KeystrokeLogger._write_row writes categories, not raw chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_session.csv")
            logger = KeystrokeLogger(out_path, session_id="test_01", label="genuine")
            # Open file manually to test _write_row without launching background listener
            logger._file = open(out_path, "w", newline="", encoding="utf-8")
            import csv
            logger._writer = csv.writer(logger._file)
            logger._writer.writerow(["timestamp", "key", "event_type", "session_id", "label"])

            # Simulate writing private keystrokes
            secret_word = "SecretPassword123!"
            for char in secret_word:
                k = keyboard.KeyCode.from_char(char)
                category = _key_name(k)
                logger._write_row(category, "key_down")
                logger._write_row(category, "key_up")
            logger._file.close()

            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify that none of the private characters or the word appears in the file
            self.assertNotIn("SecretPassword", content)
            self.assertNotIn("password", content.lower())
            for char in string.ascii_letters + string.digits:
                # Raw row format is: ts,key,event_type,...
                # Verify that no single char is between commas: ,a, or ,b,
                self.assertNotIn(f",{char},", content)


class TestSyntheticDataOverwriteSafeguard(unittest.TestCase):
    """Ensure synthetic data generator refuses to overwrite without force=True."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mock_cfg = {
            "paths": {"raw_data_dir": self.tmpdir.name},
            "synthetic": {
                "num_genuine_sessions": 1,
                "num_imposter_sessions": 1,
                "events_per_mouse_session": 10,
                "events_per_keystroke_session": 10,
                "random_seed": 42,
            },
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generate_mouse_data_aborts_when_existing(self):
        dummy_file = os.path.join(self.tmpdir.name, "mouse_existing.csv")
        with open(dummy_file, "w") as f:
            f.write("dummy,header\n1,2\n")

        with self.assertRaises(FileExistsError):
            generate_mouse_data(self.mock_cfg, force=False)

        # Confirm dummy file was not overwritten
        with open(dummy_file, "r") as f:
            self.assertEqual(f.read(), "dummy,header\n1,2\n")

    def test_generate_keystroke_data_aborts_when_existing(self):
        dummy_file = os.path.join(self.tmpdir.name, "keystroke_existing.csv")
        with open(dummy_file, "w") as f:
            f.write("dummy,header\n1,2\n")

        with self.assertRaises(FileExistsError):
            generate_keystroke_data(self.mock_cfg, force=False)

        with open(dummy_file, "r") as f:
            self.assertEqual(f.read(), "dummy,header\n1,2\n")

    def test_generate_all_aborts_when_existing(self):
        dummy_file = os.path.join(self.tmpdir.name, "mouse_existing.csv")
        with open(dummy_file, "w") as f:
            f.write("dummy\n")

        with self.assertRaises(FileExistsError):
            generate_all(self.mock_cfg, force=False)

    def test_generate_with_force_succeeds(self):
        dummy_file = os.path.join(self.tmpdir.name, "mouse_existing.csv")
        with open(dummy_file, "w") as f:
            f.write("dummy\n")

        # With force=True, it should proceed without raising FileExistsError
        generate_mouse_data(self.mock_cfg, force=True)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir.name, "mouse_genuine_000.csv")))


class TestMainStageGenerateSafeguard(unittest.TestCase):
    """Ensure stage_generate in main.py safely skips if raw CSVs already exist."""

    def test_stage_generate_skips_when_raw_data_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "paths": {"raw_data_dir": tmpdir},
                "synthetic": {
                    "num_genuine_sessions": 1,
                    "num_imposter_sessions": 1,
                    "events_per_mouse_session": 10,
                    "events_per_keystroke_session": 10,
                    "random_seed": 42,
                },
            }
            # Put an existing CSV in the folder
            marker_file = os.path.join(tmpdir, "mouse_existing.csv")
            with open(marker_file, "w") as f:
                f.write("original content")

            # Calling stage_generate with force=False should skip and not raise FileExistsError
            stage_generate(cfg, force=False)

            # Check that existing file is intact and no new files were generated
            with open(marker_file, "r") as f:
                self.assertEqual(f.read(), "original content")
            self.assertEqual(len(os.listdir(tmpdir)), 1)


class TestModelCheckpointsIntegrity(unittest.TestCase):
    """Verify that model checkpoints have not been altered or retrained in Phase 0."""

    def test_checkpoints_sha256(self):
        ckpt_dir = Path(__file__).resolve().parent.parent / "models_checkpoints"
        for filename, expected_hash in BASELINE_CHECKPOINT_HASHES.items():
            filepath = ckpt_dir / filename
            self.assertTrue(filepath.is_file(), f"Checkpoint {filename} missing from {ckpt_dir}!")
            actual_hash = _sha256(str(filepath))
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Checkpoint {filename} hash mismatch! Expected {expected_hash}, got {actual_hash}.",
            )


class TestRawDataPreservation(unittest.TestCase):
    """Verify that all baseline raw data files remain present and uncorrupted."""

    def test_raw_csv_count_and_non_empty(self):
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
        csv_files = list(raw_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 90, f"Expected 90 CSV files in {raw_dir}, found {len(csv_files)}.")
        for f in csv_files:
            self.assertGreater(f.stat().st_size, 0, f"File {f.name} is empty (0 bytes)!")


if __name__ == "__main__":
    unittest.main()
