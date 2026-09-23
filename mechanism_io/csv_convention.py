"""mechanism_io/csv_convention.py

CSV conventions for input and output files.

Two conventions are supported:

- International (default): comma as column
  separator, dot as decimal separator.
- German: semicolon as column separator, comma as
  decimal separator.

Because every CSV file handled by this project
contains a header line, the convention of a file can
be detected from its first line: when the first line
contains a semicolon, the German convention is used.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CsvConvention:
    """
    Column and decimal separators of a CSV file.

    ``delimiter`` is the column separator,
    ``decimal`` the decimal separator.
    """

    delimiter: str
    decimal: str

    @property
    def is_german(self) -> bool:
        """
        True when this is the German convention.
        """
        return self.delimiter == ";"

    def parse_float(
        self,
        value: str,
    ) -> float:
        """
        Parse a floating point number written in this
        convention.
        """
        if self.decimal != ".":
            value = value.replace(
                self.decimal,
                ".",
            )
        return float(value)

    def format_float(
        self,
        value: float,
    ) -> str:
        """
        Format a floating point number in this
        convention.

        Values that are no finite floats (``int``
        counts as finite here) are formatted with
        ``str``; booleans are rejected because they
        are not meaningful CSV numbers.
        """
        if isinstance(value, bool):
            raise TypeError(
                "booleans are not CSV numbers."
            )
        if isinstance(value, int):
            return str(value)
        text = repr(float(value))
        if self.decimal != ".":
            text = text.replace(
                ".",
                self.decimal,
            )
        return text


#: International convention (comma, dot).
INTERNATIONAL = CsvConvention(
    delimiter=",",
    decimal=".",
)

#: German convention (semicolon, comma).
GERMAN = CsvConvention(
    delimiter=";",
    decimal=",",
)


def detect_convention(
    path: str | Path,
) -> CsvConvention:
    """
    Detect the CSV convention of a file.

    When the first (header) line contains a
    semicolon, the German convention is returned,
    otherwise the international convention.
    """
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        first_line = file.readline()
    if ";" in first_line:
        return GERMAN
    return INTERNATIONAL


def sniff_delimiter(
    first_line: str,
) -> str:
    """
    Return the column delimiter of a header line.

    A semicolon in the header selects the German
    convention, otherwise the international comma is
    used.
    """
    if ";" in first_line:
        return ";"
    return ","


def open_reader(
    file: io.TextIOBase,
    *,
    convention: CsvConvention,
) -> csv.DictReader:
    """
    Create a DictReader for the given convention.
    """
    return csv.DictReader(
        file,
        delimiter=convention.delimiter,
    )


def open_writer(
    file: io.TextIOBase,
    *,
    convention: CsvConvention,
) -> csv.writer:
    """
    Create a csv writer for the given convention.
    """
    return csv.writer(
        file,
        delimiter=convention.delimiter,
    )
