import datetime
from typing import Any

import attr
import polars as pl
from treeno.base import GenericSql
from treeno.datatypes.conversions import FLOAT_TYPES, NUMERIC_TYPES
from treeno.expression import (
    Add,
    And,
    Divide,
    Equal,
    Field,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Literal,
    Minus,
    Multiply,
    Not,
    Or,
    RowConstructor,
    Value,
)
from treeno.functions.datetime import Date
from treeno.visitor import TreenoVisitor


@attr.s
class ConstexprVisitor(TreenoVisitor[Any]):
    """Constexpr visitor is used to eagerly express the results of a literal
    ValuesQuery in terms of a pl.DataFrame, which is an in-memory dataframe.

    This is different than the ExpressionVisitor in that ExpressionVisitor
    returns strictly lazy pl.Exprs.
    """

    def visit(self, node: GenericSql) -> pl.Expr:
        assert isinstance(
            node, Value
        ), "Const expression visitor only visits SQL values"
        return super().visit(node)

    def visit_Literal(self, node: Literal) -> Any:
        return node.value

    def visit_Add(self, node: Add) -> Any:
        return self.visit(node.left) + self.visit(node.right)

    def visit_Minus(self, node: Minus) -> Any:
        return self.visit(node.left) - self.visit(node.right)

    def visit_And(self, node: And) -> Any:
        # Note we're using logical operators here
        return self.visit(node.left) and self.visit(node.right)

    def visit_Multiply(self, node: Multiply) -> Any:
        return self.visit(node.left) * self.visit(node.right)

    def visit_Date(self, node: Date) -> Any:
        val = self.visit(node.value)
        if isinstance(val, str):
            return datetime.datetime.strptime(val, "%Y-%m-%d").date()
        else:
            raise NotImplementedError(
                f"No constexpr conversion to datetime from {val}"
            )

    def visit_Divide(self, node: Divide) -> Any:
        # We need to check whether the type is integral or not before
        # performing the right division operation.
        dtypes = [node.left.data_type.type_name, node.right.data_type.type_name]
        assert all(
            dtype in NUMERIC_TYPES for dtype in dtypes
        ), f"Constexpr division unsupported for types {dtypes}"
        left = self.visit(node.left)
        right = self.visit(node.right)
        if set(dtypes) & FLOAT_TYPES:
            return left / right
        # Otherwise, it must be an integral type and we want an integral result.
        else:
            return left // right

    def visit_Or(self, node: Or) -> Any:
        # Note we're using logical operators here
        return self.visit(node.left) or self.visit(node.right)

    def visit_Not(self, node: Not) -> Any:
        # Note we're using logical operators here
        return not self.visit(node.value)

    def visit_Equal(self, node: Equal) -> Any:
        return self.visit(node.left) == self.visit(node.right)

    def visit_GreaterThan(self, node: GreaterThan) -> Any:
        return self.visit(node.left) > self.visit(node.right)

    def visit_GreaterThanOrEqual(self, node: GreaterThanOrEqual) -> Any:
        return self.visit(node.left) >= self.visit(node.right)

    def visit_LessThan(self, node: LessThan) -> Any:
        return self.visit(node.left) < self.visit(node.right)

    def visit_LessThanOrEqual(self, node: LessThanOrEqual) -> Any:
        return self.visit(node.left) <= self.visit(node.right)

    def visit_RowConstructor(self, node: RowConstructor) -> Any:
        return [self.visit(val) for val in node.values]

    def visit_Field(self, node: Field) -> Any:
        raise ValueError(
            "Found dynamic field during constexpr evaluation. This shouldn't happen."
        )
