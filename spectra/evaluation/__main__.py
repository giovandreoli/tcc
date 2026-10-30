"""``python -m spectra.evaluation``: print the evaluation tables as Markdown."""

from spectra.evaluation import markdown_report

if __name__ == "__main__":
    print(markdown_report())
