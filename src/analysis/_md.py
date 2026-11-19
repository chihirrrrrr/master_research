"""DataFrame を Markdown の表にする(tabulate を使わない)。"""
import pandas as pd


def table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in d.itertuples(index=False))
