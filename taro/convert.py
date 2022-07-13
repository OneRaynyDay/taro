from typing import Dict

import polars as pl
from treeno.builder.convert import expression_from_sql, query_from_sql

from taro.expression import ExpressionVisitor
from taro.query import QueryVisitor


def to_lazyframe(sql: str, tables: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Construct a pyarrow Table from arbitrary SQL relation expressions."""
    query = query_from_sql(sql)
    return QueryVisitor.with_tables(tables).visit(query)


def to_expression(sql: str) -> pl.Expr:
    """Construct from an arbitrary SQL expression to an arrow expression.
    Used in generating predicate pushdowns, projections, and additional filters.
    """
    value = expression_from_sql(sql)
    return ExpressionVisitor().visit(value)
