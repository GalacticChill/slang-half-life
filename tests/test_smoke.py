import pytest

import slang_half_life
from slang_half_life import cli


def test_version():
    assert slang_half_life.__version__


def test_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "slang-half-life" in capsys.readouterr().out


def test_full_report_from_shipped_data(capsys, tmp_path):
    cli.main(["--assets-dir", str(tmp_path)])
    out = capsys.readouterr().out
    for heading in ("1. Is slang dying faster?", "2. Shape types", "3. Insiders first?", "4. Robustness"):
        assert heading in out
    assert len(list(tmp_path.glob("*.png"))) == 4
