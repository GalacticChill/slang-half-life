import slang_half_life
from slang_half_life import cli


def test_version():
    assert slang_half_life.__version__ == "0.1.0"


def test_cli_runs(capsys):
    cli.main([])
    assert "slang-half-life" in capsys.readouterr().out
