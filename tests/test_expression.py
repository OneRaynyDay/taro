import unittest

import polars as pl
from treeno.expression import Field, wrap_literal

from taro.expression import ExpressionVisitor


def assert_equal_expr(expr1, expr2):
    # TODO: Currently there's no good way to test equality of the underlying expression tree
    # https://github.com/pola-rs/polars/issues/4051
    return str(expr1) == str(expr2)


class ExpressionTest(unittest.TestCase):
    def setUp(self):
        self.visitor = ExpressionVisitor()

    def test_literal(self):
        assert_equal_expr(self.visitor.visit(wrap_literal(3)), pl.lit(3))
        assert_equal_expr(self.visitor.visit(wrap_literal(4.5)), pl.lit(4.5))
        assert_equal_expr(
            self.visitor.visit(wrap_literal("foo")), pl.lit("foo")
        )

    def test_field(self):
        assert_equal_expr(self.visitor.visit(Field("x")), pl.col("x"))
        # This should raise a warning for now, but we can't assign a table
        # for cols
        assert_equal_expr(
            self.visitor.visit(Field("x", table="foo")), pl.col("x")
        )

    def test_operators(self):
        # This currently is the only behavior (no floordiv), which isn't ideal.
        # Refer to the TODO in visitor.visit_Divide for more info
        assert_equal_expr(
            self.visitor.visit(wrap_literal(3) / wrap_literal(5)),
            pl.lit(3) / pl.lit(5),
        )
        assert_equal_expr(
            self.visitor.visit(wrap_literal(3) * wrap_literal(5)),
            pl.lit(3) * pl.lit(5),
        )
        assert_equal_expr(
            self.visitor.visit(wrap_literal(3) + wrap_literal(5)),
            pl.lit(3) + pl.lit(5),
        )
        assert_equal_expr(
            self.visitor.visit(wrap_literal(3) - wrap_literal(5)),
            pl.lit(3) - pl.lit(5),
        )
