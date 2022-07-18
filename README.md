# Taro

Taro is a ANSI SQL layer over [pola.rs](https://github.com/pola-rs/polars). Taro aims to be a query engine over parquet/arrow files and allows users to efficiently query collections of tabular, columnar data. **This project is currently in development and new features are added frequently.** There are a lot of hacks in this project to get a basic set of SQL operations working (so don't be surprised when there are bugs!)

# Install

```
$ pip install taro
```

# Quick Start

We can query a collection of `pl.LazyFrame`s with SQL. Here's some setup:

```python
import polars as pl
from taro.convert import to_lazyframe

table = pl.DataFrame({
    "name": ["John", "Jane", "Sephiroth"],
    "birthday": ["1997-01-01", "1997-01-01", "1998-01-03"],
    "pets": [1,5,3],
})
table = table.with_column(pl.col("birthday").str.strptime(pl.Date, fmt="%Y-%m-%d")).lazy()
```

We support equality, pure-field based inner, left, outer and cross joins. We can also generate a
literal `pl.LazyFrame` from a values table.

```python
to_lazyframe("""
SELECT *
  FROM tbl LEFT JOIN
       (VALUES
           ('Jane', DATE('2022-01-01')),
           ('Sephiroth', DATE('2022-01-02'))
       ) "employees" ("name", "join_date")
       ON tbl.name = employees.name
""", {"tbl": table}).collect()

┌───────────┬────────────┬──────┬────────────┐
│ name      ┆ birthday   ┆ pets ┆ join_date  │
│ ---       ┆ ---        ┆ ---  ┆ ---        │
│ str       ┆ date       ┆ i64  ┆ date       │
╞═══════════╪════════════╪══════╪════════════╡
│ John      ┆ 1997-01-01 ┆ 1    ┆ null       │
├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Jane      ┆ 1997-01-01 ┆ 5    ┆ 2022-01-01 │
├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Sephiroth ┆ 1998-01-03 ┆ 3    ┆ 2022-01-02 │
└───────────┴────────────┴──────┴────────────┘
```

Other common SQL operations such as CTE's, GROUPBY and ORDERBY are also supported:

```python
to_lazyframe("""
  WITH view AS (
       SELECT birthday, LOWER("name") "lower_name", pets FROM tbl)
SELECT COUNT(*) "num_people", SUM("pets") "num_pets",
       ARBITRARY("lower_name"), birthday
  FROM view
 GROUP BY birthday
""", {"tbl": table.lazy()}).collect()

┌────────────┬────────────┬──────────┬────────────┐
│ birthday   ┆ num_people ┆ num_pets ┆ lower_name │
│ ---        ┆ ---        ┆ ---      ┆ ---        │
│ date       ┆ u32        ┆ i64      ┆ str        │
╞════════════╪════════════╪══════════╪════════════╡
│ 1998-01-03 ┆ 1          ┆ 3        ┆ sephiroth  │
├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 1997-01-01 ┆ 2          ┆ 6        ┆ john       │
└────────────┴────────────┴──────────┴────────────┘
```
