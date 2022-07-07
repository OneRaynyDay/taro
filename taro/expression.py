"""
Taro aims to take Treeno AST and convert it into arrow expressions to
perform predicate pushdown, filtering, projection, etc.

Here we'll provide some basic helper functions to convert arbitrary SQL to
arrow expressions.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeVar

import attr
import pyarrow as pa
import pyarrow.dataset as ds
import treeno.datatypes.types as tt
from treeno.base import GenericSql, PrintOptions
from treeno.builder.convert import expression_from_sql, query_from_sql
from treeno.expression import (
    Add,
    AliasedValue,
    And,
    Cast,
    Divide,
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
    Value,
)
from treeno.functions.aggregate import (
    AggregateFunction,
    Arbitrary,
    Avg,
    BoolAnd,
    BoolOr,
    Count,
    Every,
    Max,
    Min,
    StdDev,
    Sum,
    UnaryAggregateFunction,
    Variance,
)
from treeno.groupby import GroupBy, GroupingSet
from treeno.relation import SelectQuery, Table
from treeno.visitor import TreenoVisitor

GenericAggregateFunction = TypeVar(
    "GenericAggregateFunction", bound=AggregateFunction
)

_TIME_PRECISION_MAP = {
    0: "s",
    3: "ms",
    6: "us",
    9: "ns",
}

# List of unary aggregation fns to their names in arrow
_AGGREGATION_FN_TO_NAME = {
    Sum: "sum",
    Arbitrary: "one",
    Min: "min",
    Max: "max",
    Avg: "mean",
    # TODO: Check if this is StdDevPop in their implementation
    StdDev: "stddev",
    Variance: "variance",
    BoolAnd: "all",
    # Include BoolAnd's alias - Every
    Every: "all",
    BoolOr: "any",
    # TODO: count_distinct will be a bit harder to implement - we need to check whether
    # Count contains a DISTINCT clause inside
    Count: "count",
}


def _time_precision_to_unit(precision: int) -> str:
    if precision not in _TIME_PRECISION_MAP:
        raise NotImplementedError(
            f"Precision {precision} does not have a perfect arrow unit mapping"
        )
    return _TIME_PRECISION_MAP(precision)


def _require_64bit_time(unit: str) -> bool:
    assert unit in set(_TIME_PRECISION_MAP.values())
    return unit in {"us", "ns"}


def to_arrow_datatype(data_type: tt.DataType) -> pa.DataType:
    # TODO: Let's replace this with structural pattern matching when
    # we bump to python 3.10+!
    if data_type.type_name == tt.BOOLEAN:
        return pa.bool_()
    if data_type.type_name == tt.INTEGER:
        return pa.int32()
    if data_type.type_name == tt.TINYINT:
        return pa.int8()
    if data_type.type_name == tt.SMALLINT:
        return pa.int16()
    if data_type.type_name == tt.BIGINT:
        return pa.int64()
    if data_type.type_name == tt.REAL:
        return pa.float32()
    if data_type.type_name == tt.DOUBLE:
        return pa.float64()
    if data_type.type_name == tt.DECIMAL:
        # TODO: Verify 128 is the same size on trino
        return pa.decimal128(
            precision=data_type.parameters["precision"],
            scale=data_type.parameters["scale"],
        )
    # VARCHAR and friends can have max_chars limit, but arrow doesn't
    if data_type.type_name in (tt.VARCHAR, tt.CHAR):
        return pa.string()
    # Trino doesn't allow max length, but arrow does
    if data_type.type_name == tt.VARBINARY:
        return pa.binary(length=-1)
    if data_type.type_name == tt.TIME:
        assert (
            "precision" in data_type.parameters
        ), f"Time type must have precision defined, got {data_type.sql(PrintOptions())}"
        assert (
            "timezone" not in data_type.parameters
        ), f"Arrow doesn't have a tz-aware time type, got {data_type.sql(PrintOptions())}"
        unit = _time_precision_to_unit(data_type.parameters["precision"])
        if _require_64bit_time(unit):
            return pa.time64(unit)
        else:
            return pa.time32(unit)
    if data_type.type_name == tt.TIMESTAMP:
        assert (
            "precision" in data_type.parameters
        ), f"Timestamp type must have precision defined, got {data_type.sql(PrintOptions())}"
        assert "timezone" not in data_type.parameters, (
            "Arrow's timestamp type requires a fixed timezone(e.g. 'America/New_York'), but "
            "Trino's timestamp type can have dynamic timezone information, "
            f"got {data_type.sql(PrintOptions())}"
        )
        return pa.timestamp(
            _time_precision_to_unit(data_type.parameters["precision"])
        )
    raise NotImplementedError(
        f"Data type conversion for {data_type.sql(PrintOptions())} to arrow is not supported"
    )


@attr.s
class ArrowExpressionVisitor(TreenoVisitor[ds.Expression]):
    def visit(self, node: GenericSql) -> ds.Expression:
        assert isinstance(
            node, Value
        ), "Arrow expression visitor only visits SQL values"
        return super().visit(node)

    def visit_Literal(self, node: Literal) -> ds.Expression:
        return ds.scalar(node.value)

    def visit_Add(self, node: Add) -> ds.Expression:
        return self.visit(node.left) + self.visit(node.right)

    def visit_Minus(self, node: Minus) -> ds.Expression:
        return self.visit(node.left) - self.visit(node.right)

    def visit_And(self, node: And) -> ds.Expression:
        return self.visit(node.left) & self.visit(node.right)

    def visit_Multiply(self, node: Multiply) -> ds.Expression:
        return self.visit(node.left) * self.visit(node.right)

    def visit_Divide(self, node: Divide) -> ds.Expression:
        return self.visit(node.left) / self.visit(node.right)

    def visit_Or(self, node: Or) -> ds.Expression:
        return self.visit(node.left) | self.visit(node.right)

    def visit_Not(self, node: Not) -> ds.Expression:
        return ~self.visit(node.value)

    def visit_Equal(self, node: Equal) -> ds.Expression:
        return self.visit(node.left) == self.visit(node.right)

    def visit_GreaterThan(self, node: GreaterThan) -> ds.Expression:
        return self.visit(node.left) > self.visit(node.right)

    def visit_GreaterThanOrEqual(
        self, node: GreaterThanOrEqual
    ) -> ds.Expression:
        return self.visit(node.left) >= self.visit(node.right)

    def visit_LessThan(self, node: LessThan) -> ds.Expression:
        return self.visit(node.left) < self.visit(node.right)

    def visit_LessThanOrEqual(self, node: LessThanOrEqual) -> ds.Expression:
        return self.visit(node.left) <= self.visit(node.right)

    def visit_Cast(self, node: Cast) -> ds.Expression:
        return self.visit(node.expr).cast(to_arrow_datatype(node.data_type))

    def visit_IsNull(self, node: IsNull) -> ds.Expression:
        return self.visit(node.value).is_null()

    def visit_InList(self, node: InList) -> ds.Expression:
        values = [self.visit(val) for val in node.exprs]
        return self.visit(node.value).isin(values)

    def visit_Field(self, node: Field) -> ds.Expression:
        assert (
            not node.table
        ), "Currently visitor cannot support multi-table fields. They are all assumed to be from the same table."
        return ds.field(node.name)

    def visit_AliasedValue(self, node: AliasedValue) -> ds.Expression:
        # We don't do anything w/ the aliasing - still return the same expression
        return self.visit(node.value)


@attr.s
class ArrowQueryVisitor(TreenoVisitor[pa.Table]):
    tables: Dict[str, pa.Table] = attr.ib()
    expr_visitor: ArrowExpressionVisitor = attr.ib(
        init=False, factory=ArrowExpressionVisitor
    )
    active_ctes: Dict[str, pa.Table] = attr.ib(init=False, factory=dict)

    def get_aggregate_projection(
        self, val: Value, grouped_keys: List[str]
    ) -> Tuple[str, Optional[Tuple[str, str]]]:
        """Checks for an arbitrary expression whether it's an aggregate expression or not.
        TODO: Currently we assume there is only a top-level aggregate function involving a single field, since
        that's what arrow can provide as of v8.0.0.

        Return a tuple of (alias name, (key name, aggregation fn name)) where:
        1. Alias name is the name we assigned the aggregation
        2. Key name is the name of the column we'd like to aggregate over
        3. Aggregation fn name is the name of the corresponding aggregation fn we'd like to run

        For special cases like selecting the groups by field directly, (2, 3) above are replaced with a None
        """
        # Special case: If the field is grouped on, it is considered an aggregate expression
        if isinstance(val, Field):
            assert val.name in grouped_keys, (
                "Field expressions are only valid in a groupby setting if it is being grouped by directly. "
                f"Expected a field in {grouped_keys}, got {val.name} instead"
            )
            return (val.name, None)

        assert isinstance(
            val, AliasedValue
        ), f"Aggregate expressions must have an alias for output column name, got {val.sql(PrintOptions())} instead"
        # This will be the output column name. Unfortunately arrow doesn't allow us to pass in a name for the aggregation result as of v8.0.0
        alias_name = val.alias
        val = val.value
        assert isinstance(
            val, AggregateFunction
        ), f"In the presence of a GROUP BY only aggregate expressions are allowed, got {val.sql(PrintOptions())}"
        assert isinstance(
            val, UnaryAggregateFunction
        ), "Arrow currently only supports unary aggregate functions"
        assert isinstance(
            val.value, Field
        ), f"Arrow currently only supports aggregations on pure fields, not expressions on fields like {val.value.sql(PrintOptions())}"
        assert (
            type(val) in _AGGREGATION_FN_TO_NAME
        ), f"Cannot translate aggregation function {type(val).__name__} to arrow"
        agg_fn_name = _AGGREGATION_FN_TO_NAME[type(val)]
        key_name = val.value.name
        return (alias_name, (key_name, agg_fn_name))

    def get_non_aggregate_projection(
        self, val: Value
    ) -> Tuple[str, ds.Expression]:
        """For projections, get the values and assign them to names"""
        # TODO: assert there are no aggregation functions deep within the expression.
        if isinstance(val, AliasedValue):
            return (val.alias, self.expr_visitor.visit(val))
        if isinstance(val, Field):
            return (val.name, self.expr_visitor.visit(val))
        raise NotImplementedError(
            f"Projected value of type {val.__class__.__name__} does not have a well-defined column name"
        )

    def get_groupby_keys(self, node: GroupBy) -> List[str]:
        """Currently arrow supports very basic groupby functionalities.
        As of pyarrow v8.0.0, We can only get the pure fields from a given table.
        """
        columns = []
        # TODO: Perhaps allow for empty groupbys by taking all columns?
        assert len(node.groups) == 1
        group = node.groups[0]
        assert isinstance(group, GroupingSet)
        for val in group.values:
            assert isinstance(
                val, Field
            ), f"Arrow only supports groupby directly on fields, got {val.sql(PrintOptions())} instead."
            columns.append(val.name)
        return columns

    def visit_Table(self, node: Table) -> pa.Table:
        tbl_name = node.name
        # When there is an actual table AND a CTE with the same name, the CTE is prioritized.
        # Here, x is an actual table with a field date, but we picked the CTE.
        # trino:default> WITH f (a) AS (SELECT 1), x (date) AS (SELECT f.a FROM f) SELECT x.date FROM x;
        # date
        # ------
        # 1
        # (1 row)
        if tbl_name in self.active_ctes:
            return self.active_ctes[tbl_name]

        if tbl_name not in self.tables:
            raise KeyError(
                f"Failed to find table {tbl_name} in supplied list of tables: {list(self.tables.keys())}"
            )
        return self.tables[tbl_name]

    def visit_SelectQuery(self, node: SelectQuery) -> pa.Table:
        assert (
            node.from_ is not None
        ), "Empty FROM queries not yet supported. This is a constexpr table."
        assert node.window is None, "Window functions currently not supported."
        # The CTE's defined in this query will go out of scope afterwards, including overridden table names
        old_ctes = self.active_ctes.copy()

        for relation in node.with_:
            # For a sequence of CTE's defined, one can use the previously defined CTE's
            # at any point:
            # trino> WITH f (a) AS (SELECT 1), g (b) AS (SELECT f.a FROM f) SELECT g.b FROM g;
            # b
            # ---
            # 1
            # (1 row)
            self.active_ctes[relation.alias] = self.visit(relation)
        input_table = self.visit(node.from_)
        table: pa.Table
        # So there are some subtleties about the projections here regarding groupby
        # Currently all aggregation functions specified by arrow are in this list:
        # https://arrow.apache.org/docs/python/compute.html#grouped-aggregations
        # which in my opinion is really strangely defined - why string names
        # for the aggregations as opposed to just pyarrow expressions?
        #
        # But anyways, we'll do the following if we are grouping:
        # 1. Figure out the groupby keys
        # 2. Get all the aggregate functions (in field name, fn name form)
        # 3. Apply the groupby after eager to_table on the dataset
        #
        # If we're not grouping, we'll do the following:
        # 1. Check there are no aggregate functions and get all projections
        # 2. Apply projections in to_table
        if node.groupby:
            groupby_keys = self.get_groupby_keys(node.groupby)
            aggregations = []
            rename_to_alias = defaultdict(list)
            for val in node.select:
                alias, fn_args = self.get_aggregate_projection(
                    val, groupby_keys
                )

                # Don't perform an aggregation, as we are selecting the columns directly
                if fn_args is None:
                    rename_to_alias[alias].append(alias)
                    continue

                key_name, fn_name = fn_args
                arrow_formatted_agg_column = f"{key_name}_{fn_name}"
                # Although smelly, having two identical aggs named differently is technically legal
                rename_to_alias[arrow_formatted_agg_column].append(alias)
                aggregations.append(fn_args)
            table = input_table.group_by(groupby_keys).aggregate(aggregations)

            new_column_names = []
            for name in table.column_names:
                # TODO: Technically we can use a deque here and make this faster
                assert len(rename_to_alias[name])
                new_column_names.append(rename_to_alias[name].pop(0))
            table = table.rename_columns(new_column_names)
        else:
            projections = dict(
                self.get_non_aggregate_projection(val) for val in node.select
            )
            filter_predicate = (
                self.expr_visitor.visit(node.where) if node.where else None
            )
            table = _table_to_dataset(input_table).to_table(
                columns=projections, filter=filter_predicate
            )
        self.active_ctes = old_ctes
        return table


def arrow_from_query(sql: str, tables: Dict[str, pa.Table]) -> ds.Dataset:
    """Construct a pyarrow Table from arbitrary SQL relation expressions."""
    query = query_from_sql(sql)
    return ArrowQueryVisitor(tables=tables).visit(query)


def arrow_from_expression(sql: str) -> ds.Expression:
    """Construct from an arbitrary SQL expression to an arrow expression.
    Used in generating predicate pushdowns, projections, and additional filters.
    """
    value = expression_from_sql(sql)
    return ArrowExpressionVisitor().visit(value)


def _table_to_dataset(tbl: pa.Table) -> ds.Dataset:
    return ds.dataset([tbl])
