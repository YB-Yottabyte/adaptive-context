from unittest.mock import patch

from baseline import main


def test_main_delegates_to_cli_application() -> None:
    with patch("baseline.BaselineCLI") as cli_class:
        main()

    cli_class.return_value.run.assert_called_once_with()
