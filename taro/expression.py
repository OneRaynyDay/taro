"""
Taro aims to take Treeno AST and convert it into arrow expressions to
perform predicate pushdown, filtering, projection, etc.

Here we'll provide some basic helper functions to convert arbitrary SQL to
arrow expressions.
"""

import logging
from typing import Optional, TypeVar

import attr
import polars as pl
from treeno.base import GenericSql, PrintOptions
from treeno.expression import (
    Add,
    AliasedValue,
    And,
    Case,
    Cast,
    Divide,
    Else,
    Equal,
    Field,
    GreaterThan,
    GreaterThanOrEqual,
    InList,
    IsNull,
    LessThan,
    LessThanOrEqual,
    Literal,
    Minus,
    Multiply,
    Not,
    Or,
    Star,
    Value,
    When,
)
from treeno.functions.aggregate import (
    AggregateFunction,
    Arbitrary,
    Avg,
    BoolAnd,
    BoolOr,
    Count,
    CountIf,
    Every,
    Max,
    Min,
    Sum,
)
from treeno.functions.string import Length, Lower, LTrim, RTrim
from treeno.visitor import TreenoVisitor

from taro.datatype import to_polars_datatype

GenericAggregateFunction = TypeVar(
    "GenericAggregateFunction", bound=AggregateFunction
)
LOGGER = logging.getLogger(__name__)


@attr.s
class ExpressionVisitor(TreenoVisitor[pl.Expr]):
    def visit(self, node: GenericSql) -> pl.Expr:
        assert isinstance(
            node, Value
        ), "Arrow expression visitor only visits SQL values"
        return super().visit(node)

    def visit_Literal(self, node: Literal) -> pl.Expr:
        return pl.lit(node.value)

    def visit_Add(self, node: Add) -> pl.Expr:
        return self.visit(node.left) + self.visit(node.right)

    def visit_Minus(self, node: Minus) -> pl.Expr:
        return self.visit(node.left) - self.visit(node.right)

    def visit_And(self, node: And) -> pl.Expr:
        return self.visit(node.left) & self.visit(node.right)

    def visit_Multiply(self, node: Multiply) -> pl.Expr:
        return self.visit(node.left) * self.visit(node.right)

    def visit_Divide(self, node: Divide) -> pl.Expr:
        # TODO: This is hard to map to polars because we don't
        # necessarily know the type of the inputs during graph
        # construction time, which means we can't determine whether
        # we should floordiv or truediv.
        return self.visit(node.left) / self.visit(node.right)

    def visit_Or(self, node: Or) -> pl.Expr:
        return self.visit(node.left) | self.visit(node.right)

    def visit_Not(self, node: Not) -> pl.Expr:
        return ~self.visit(node.value)

    def visit_Equal(self, node: Equal) -> pl.Expr:
        return self.visit(node.left) == self.visit(node.right)

    def visit_GreaterThan(self, node: GreaterThan) -> pl.Expr:
        return self.visit(node.left) > self.visit(node.right)

    def visit_Star(self, node: Star) -> pl.Expr:
        assert (
            not node.table
        ), "Table-granular star expressions not supported in polars"
        return pl.col("*")

    def visit_GreaterThanOrEqual(self, node: GreaterThanOrEqual) -> pl.Expr:
        return self.visit(node.left) >= self.visit(node.right)

    def visit_LessThan(self, node: LessThan) -> pl.Expr:
        return self.visit(node.left) < self.visit(node.right)

    def visit_LessThanOrEqual(self, node: LessThanOrEqual) -> pl.Expr:
        return self.visit(node.left) <= self.visit(node.right)

    def visit_Cast(self, node: Cast) -> pl.Expr:
        return self.visit(node.expr).cast(to_polars_datatype(node.data_type))

    def visit_IsNull(self, node: IsNull) -> pl.Expr:
        return self.visit(node.value).is_null()

    def visit_InList(self, node: InList) -> pl.Expr:
        values = [self.visit(val) for val in node.exprs]
        return self.visit(node.value).is_in(values)

    def visit_Field(self, node: Field) -> pl.Expr:
        if node.table:
            LOGGER.warn(
                f"Currently visitor cannot support multi-table fields - assuming {node.sql(PrintOptions())} is inferrable without table."
            )
        return pl.col(node.name)

    def visit_AliasedValue(self, node: AliasedValue) -> pl.Expr:
        # We don't do anything w/ the aliasing - still return the same expression
        return self.visit(node.value).alias(node.alias)

    def visit_Sum(self, node: Sum) -> pl.Expr:
        return self.visit(node.value).sum()

    def visit_Arbitrary(self, node: Arbitrary) -> pl.Expr:
        # Well... arbitrary isn't very well-defined, so I choose first
        return self.visit(node.value).first()

    def visit_Avg(self, node: Avg) -> pl.Expr:
        return self.visit(node.value).mean()

    def visit_BoolAnd(self, node: BoolAnd) -> pl.Expr:
        return self.visit(node.value).all()

    def visit_Every(self, node: Every) -> pl.Expr:
        # Every is an alias for BoolAnd
        return self.visit(node.value).all()

    def visit_BoolOr(self, node: BoolOr) -> pl.Expr:
        return self.visit(node.value).any()

    def visit_Case(self, node: Case) -> pl.Expr:
        value = self.visit(node.value)
        chained_case = None
        for branch in node.branches:
            chained_case = self.visit_When(branch, value, chained_case)
        if node.else_:
            chained_case = self.visit_Else(node.else_, chained_case)
        else:
            chained_case = chained_case.otherwise(pl.lit(None))
        return chained_case

    def visit_When(
        self, node: When, value: pl.Expr, chained_case: Optional[pl.Expr] = None
    ) -> pl.Expr:
        pred = value == self.visit(node.condition)
        if chained_case is None:
            cond = pl.when(pred)
        else:
            cond = chained_case.when(pred)
        return cond.then(self.visit(node.value))

    def visit_Else(self, node: Else, chained_case: pl.Expr) -> pl.Expr:
        return chained_case.otherwise(self.visit(node.value))

    def visit_Count(self, node: Count) -> pl.Expr:
        return self.visit(node.value).count()

    def visit_CountIf(self, node: CountIf) -> pl.Expr:
        return (
            pl.when(self.visit(node.value)).then(pl.lit(1)).otherwise(0).sum()
        )

    def visit_Max(self, node: Max) -> pl.Expr:
        # TODO: https://github.com/pola-rs/polars/issues/3997
        assert not node.num_values, "Polars currently does not support top k"
        return self.visit(node.value).max()

    def visit_Min(self, node: Min) -> pl.Expr:
        # TODO: https://github.com/pola-rs/polars/issues/3997
        assert not node.num_values, "Polars currently does not support bottom k"
        return self.visit(node.value).min()

    def visit_Length(self, node: Length) -> pl.Expr:
        return self.visit(node.string).str.lengths()

    def visit_Lower(self, node: Lower) -> pl.Expr:
        return self.visit(node.string).str.to_lowercase()

    def visit_LTrim(self, node: LTrim) -> pl.Expr:
        return self.visit(node.string).str.lstrip()

    def visit_RTrim(self, node: RTrim) -> pl.Expr:
        return self.visit(node.string).str.rstrip()
