import datetime
import unittest

import pytest
from treeno.expression import Field, RowConstructor, wrap_literal
from treeno.functions.datetime import Date

from taro.constexpr import ConstexprVisitor


class ConstExprTest(unittest.TestCase):
    def setUp(self):
        self.visitor = ConstexprVisitor()

    def test_literal(self):
        assert self.visitor.visit(wrap_literal(3)) == 3
        assert self.visitor.visit(wrap_literal(3.14)) == 3.14
        assert self.visitor.visit(wrap_literal("foo")) == "foo"

    def test_operators(self):
        assert self.visitor.visit(
            wrap_literal(3) + wrap_literal(5.3)
        ) == pytest.approx(8.3)
        assert self.visitor.visit(wrap_literal(3) / wrap_literal(5)) == 0
        assert self.visitor.visit(
            wrap_literal(3.0) / wrap_literal(5)
        ) == pytest.approx(3 / 5)
        assert (
            self.visitor.visit(wrap_literal(True) & wrap_literal(False))
            == False
        )
        assert (
            self.visitor.visit(wrap_literal(True) | wrap_literal(False)) == True
        )
        assert (
            self.visitor.visit(
                (wrap_literal(3) + wrap_literal(5)) == wrap_literal(8)
            )
            == True
        )

    def test_row_constructor(self):
        assert self.visitor.visit(
            RowConstructor([wrap_literal("foo"), wrap_literal(3)])
        ) == ["foo", 3]

    def test_date(self):
        assert self.visitor.visit(
            Date(wrap_literal("2022-01-01"))
        ) == datetime.date(2022, 1, 1)

    def test_field_should_throw(self):
        with pytest.raises(ValueError):
            self.visitor.visit(Field("x"))
