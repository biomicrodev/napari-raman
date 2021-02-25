from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from xarray import Dataset


def bin2hex(s: bytes) -> List[str]:
    lines = s.hex().split("0d0a")

    new_lines = []
    for line in lines:
        try:
            val = bytes.fromhex(line).decode("utf-8")
        except Exception:
            continue
        else:
            new_lines.append(val)

    return new_lines


def _read_spc(file: Path) -> Dict[str, str]:
    with file.open("rb") as f:
        lines = bin2hex(f.read())

    parameters = {}
    for line in lines:
        args = line.split("=")
        if len(args) != 2:
            continue
        key, value = args
        parameters[key] = value
    return parameters


def _read_txt(path: Path) -> Dataset:
    """
    Text files in wire projects contain spectra data. They are tsv files, but the
    header line is separated by more than one tab char, ostensibly to accommodate long
    floats in string format. The logic here is to deal with those extra tab chars.

    In addition, both line and grid data have X and Y columns, and although there is a
    difference in the X and Y coordinates, I can't be bothered to code that in; instead
    I'm specifying the structure manually.
    """
    df: pd.DataFrame = pd.read_csv(path, sep="\t", dtype=float)

    to_rename = {"wave": "wavenumber"}

    columns = df.columns.tolist()
    columns = [col for col in columns if not col.startswith("Unnamed: ")]
    columns = [col[1:] if col.startswith("#") else col for col in columns]
    columns = [col.lower() for col in columns]
    columns = [to_rename[col] if col in to_rename else col for col in columns]

    df = df.dropna(axis=1)
    if len(df.columns) != len(columns):
        raise RuntimeError("Unable to interpret file")

    df.columns = columns

    df.sort_values(by=list(df.columns), axis=0, ignore_index=True, inplace=True)
    if "x" in df.columns and "y" in df.columns:
        df.index = pd.MultiIndex.from_tuples(
            [
                ((df.loc[i, "x"], df.loc[i, "y"]), df.loc[i, "wavenumber"])
                for i in range(len(df))
            ],
            names=["coordinates", "wavenumber"],
        )
        df.drop(["x", "y", "wavenumber"], axis=1, inplace=True)
    else:
        df.index = pd.Index(df["wavenumber"])
        df = df.drop(labels=["wavenumber"], axis=1)

    return df.to_xarray()


@dataclass(frozen=True)
class DataSource:
    path: Path

    def __post_init__(self):
        if not self._spc_path.is_file():
            raise RuntimeError("spc file does not exist")
        if not self._txt_path.is_file():
            raise RuntimeError("txt file does not exist")

    @property
    def _spc_path(self) -> Path:
        return self.path.parent / (self.path.name + ".spc")

    @property
    def _txt_path(self) -> Path:
        return self.path.parent / (self.path.name + ".txt")

    @property
    def data(self) -> Dataset:
        attrs = _read_spc(self._spc_path)
        data = _read_txt(self._txt_path)
        data.attrs.update(attrs)
        return data
