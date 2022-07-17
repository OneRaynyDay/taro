# Taro

Taro is a ANSI SQL layer over [pola.rs](https://github.com/pola-rs/polars). Taro aims to be a query engine over parquet/arrow files and allows users to efficiently query collections of tabular, columnar data. **This project is currently in development and new features are added frequently.**

# Install

```
$ pip install taro
```

# Quick Start

To query a collection of `pl.LazyFrame`s with SQL:

```python
import taro as tr

tbl =

tr.convert.to_lazyframe("""
SELECT *
FROM tbl
LEFT JOIN (VALUES (2,'c'),(0,'d')) "t" ("w", "z")
ON x = w
""")
```
