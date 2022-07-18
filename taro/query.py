import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import attr
import polars as pl
from treeno.base import PrintOptions, SetQuantifier
from treeno.expression import Equal, Field, Value
from treeno.groupby import GroupBy, GroupingSet
from treeno.orderby import NullOrder, OrderTerm, OrderType
from treeno.relation import (
    AliasedRelation,
    Join,
    JoinConfig,
    JoinOnCriteria,
    JoinType,
    Relation,
    SelectQuery,
    Table,
    TableQuery,
    TableSample,
    Unnest,
    ValuesQuery,
)
from treeno.visitor import TreenoVisitor
from typing_extensions import Self

from taro.constexpr import ConstexprVisitor
from taro.expression import ExpressionVisitor
from taro.utils import ScopedDict

LOGGER = logging.getLogger(__name__)


# TODO: Currently polars doesn't support right joins for a pretty arbitrary reason
# https://github.com/pola-rs/polars/issues/3934. They should though, and when they do I should
# add the support here
_JOIN_TYPE_MAPPING = {
    JoinType.INNER: "inner",
    JoinType.LEFT: "left",
    JoinType.OUTER: "outer",
    JoinType.CROSS: "cross",
}


def get_expr_column_name(expr: pl.Expr) -> str:
    # TODO: This is a huge hack and should not be used in the long term. I think this has like
    # a million bugs.
    # The cols should look like: 'col("foobar")',
    # so I'll just parse it as a hack here for now while that ticket (and its linked ticket) are not
    # addressed yet.
    # The negative lookbehind expression is to ignore escaped double quotes, and
    # I assume if the column is aliased to any string the name would be at the end minus the last piece of "), hence -2.
    col_name = str(expr)
    return re.split(r'(?<!\\)"', col_name)[-2]


@attr.s
class JoinContext:
    columns: Dict[str, List[str]] = attr.ib(
        init=False, factory=lambda: defaultdict(list)
    )
    tables: List[str] = attr.ib(init=False, factory=list)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.columns = defaultdict(list)
        self.tables = []

    def add_table(self, table_name: str, columns: List[str]) -> None:
        self.tables.append(table_name)
        for col in columns:
            self.columns[col].append(table_name)


@attr.s
class QueryVisitor(TreenoVisitor[pl.LazyFrame]):
    expr_visitor: ExpressionVisitor = attr.ib(
        init=False, factory=ExpressionVisitor
    )
    constexpr_visitor: ConstexprVisitor = attr.ib(
        init=False, factory=ConstexprVisitor
    )
    active_ctes: ScopedDict[str, pl.LazyFrame] = attr.ib(
        init=False, factory=ScopedDict
    )
    active_join_context: JoinContext = attr.ib(init=False, factory=JoinContext)

    @classmethod
    def with_tables(cls, tables: Dict[str, pl.LazyFrame]) -> "QueryVisitor":
        visitor = cls()
        for k, v in tables.items():
            visitor.active_ctes[k] = v
        return visitor

    def get_groupby_exprs(self, node: GroupBy) -> List[pl.Expr]:
        """Get dynamic expressions to group by"""
        # TODO: This isn't too hard to implement for other types like Cube and GroupingSets
        assert all(
            isinstance(group, GroupingSet) and len(group.values) == 1
            for group in node.groups
        ), "Only GroupingSets with single values are allowed for now"
        return [
            self.expr_visitor.visit(group.values[0]) for group in node.groups
        ]

    def orderby_table(
        self, orderby: OrderTerm, tbl: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Given an orderby term, apply it to tbl.

        Since we're doing this in an iterative fashion, allow polars to perform optimizations to batch
        orderbys together depending on order type and null ordering.
        """
        kwargs = {}
        if orderby.order_type == OrderType.ASC:
            kwargs["reverse"] = False
        elif orderby.order_type == OrderType.DESC:
            kwargs["reverse"] = True
        else:
            raise NotImplementedError(
                f"Unsupported order type {orderby.order_type}"
            )

        if orderby.null_order != NullOrder.FIRST:
            LOGGER.warn(
                f"Null order {orderby.null_order} doesn't allow arbitrary expressions on orderby in Polars. "
                "Ignoring this field for now and defaulting to null order FIRST"
            )

        kwargs["by"] = self.expr_visitor.visit(orderby.value)
        return tbl.sort(**kwargs)

    def get_table_name(self, table: Relation) -> Optional[str]:
        if isinstance(table, Table):
            return table.name
        if isinstance(table, AliasedRelation):
            return table.alias
        return None

    def get_join_kwargs(
        self,
        config: JoinConfig,
    ) -> Dict[str, Any]:
        """Take the join configs from SQL and translate them to arguments into pl.LazyFrame.join

        Args:
            config: The join configuration which specifies which columns are being joined with and type of join
            left_table_name: The name of the left table to be used to reference in join configurations.
                If None, the relation has no name. Note that CROSS JOINs do not need the table name but
                they also don't reference the table columns.
            right_table_name: The name of the left table to be used to reference in join configurations.
                If None, the relation has no name.
        """

        def populate_config(val: Value, config: Dict[str, Any]) -> None:
            assert isinstance(
                val, Field
            ), "Polars can only directly join on fields right now"
            assert (
                val.name in self.active_join_context.columns
            ), f"Column {val.name} cannot be resolved. No such column found in join context"
            field = self.expr_visitor.visit(val)
            table_name: str
            if val.table is None:
                tbl_names = self.active_join_context.columns[val.name]
                assert (
                    len(tbl_names) == 1
                ), f"Column {val.name} cannot be resolved. Found multiple tables {tbl_names} with column name."
                table_name = tbl_names[0]
            else:
                assert isinstance(
                    (table_name := val.table), str
                ), f"Only support direct table referenced fields as of right now, got {val.sql(PrintOptions())}"
                assert (
                    table_name in self.active_join_context.columns[val.name]
                ), f"Column {val.name} cannot be resolved. Specified table {table_name} not in join context {self.active_join_context.columns[val.name]}"
            active_table_names = self.active_join_context.tables
            assert (
                table_name in active_table_names
            ), f"Table {table_name} not found in active tables: {active_table_names}"
            # The rightmost table should be the last active cte
            if table_name == active_table_names[-1]:
                config["right_on"] = field
            else:
                config["left_on"] = field

        kwargs = {}
        assert (
            config.join_type in _JOIN_TYPE_MAPPING
        ), f"No available join type in polars for trino join type {config.join_type}"
        kwargs["how"] = _JOIN_TYPE_MAPPING[config.join_type]
        assert (
            not config.natural
        ), "Natural joins are bad. Even if I can support this I won't"
        if config.criteria:
            assert isinstance(
                config.criteria, JoinOnCriteria
            ), "Using joins are bad. Even if I can support this I won't"
            constraint = config.criteria.constraint
            # TODO: There are some problems with polar's joins - https://github.com/pola-rs/polars/issues/3935
            assert isinstance(
                constraint, Equal
            ), "Polars only supports joins on equality on direct fields as of right now"
            populate_config(constraint.left, kwargs)
            populate_config(constraint.right, kwargs)
        return kwargs

    def visit_Table(self, node: Table) -> pl.LazyFrame:
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
        raise KeyError(
            f"Failed to find table {tbl_name} in supplied list of tables: {list(self.active_ctes.to_dict().keys())}"
        )

    def visit_TableQuery(self, node: TableQuery) -> pl.LazyFrame:
        return self.visit(node.table)

    def visit_ValuesQuery(self, node: ValuesQuery) -> pl.LazyFrame:
        return pl.DataFrame(
            [self.constexpr_visitor.visit(val) for val in node.exprs],
            orient="row",
        ).lazy()

    def visit_Unnest(self, node: Unnest) -> pl.LazyFrame:
        # This one is a bit hard - we can accept arbitrary Values as the array(or struct) type in SQL but I don't know
        # how to express the unnest as a node to create a LazyFrame
        raise NotImplementedError("UNNEST is currently not supported")

    def visit_TableSample(self, node: TableSample) -> pl.LazyFrame:
        # I left a ticket here:
        # https://github.com/pola-rs/polars/issues/3933
        raise NotImplementedError("TABLESAMPLE is currently not supported")

    def visit_Join(self, node: Join) -> pl.LazyFrame:
        left_table = self.visit(node.left_relation)
        right_table = self.visit(node.right_relation)

        left_name = self.get_table_name(node.left_relation)
        right_name = self.get_table_name(node.right_relation)
        if left_name:
            self.active_join_context.add_table(left_name, left_table.columns)
        if right_name:
            self.active_join_context.add_table(right_name, right_table.columns)

        join_kwargs = self.get_join_kwargs(node.config)
        # TODO: Currently Polars removes the right table join column in the output.
        # I reported it at: https://github.com/pola-rs/polars/issues/3936
        # Hopefully we can get the column back for downstream queries. For now we
        # will add the column back into the right table and then re-select it after alias
        # We also only do this if the column names are different, because otherwise we'll shadow
        # the left join column. (We don't have a right table join as of right now so the only times Nones
        # will shadow the actual column value is if it's a left join or some outer join)
        if "right_on" in join_kwargs and str(join_kwargs["right_on"]) != str(
            join_kwargs["left_on"]
        ):
            join_col = join_kwargs["right_on"]
            col_name = get_expr_column_name(join_col)
            temp_name = f"{col_name}_joined"
            right_table = right_table.with_column(join_col.alias(temp_name))
            joined_table = (
                left_table.join(right_table, **join_kwargs)
                .with_column(pl.col(temp_name).alias(col_name))
                .drop(temp_name)
            )
        else:
            joined_table = left_table.join(right_table, **join_kwargs)
        return joined_table

    def visit_AliasedRelation(self, node: AliasedRelation) -> pl.LazyFrame:
        table = self.visit(node.relation)
        # TODO: Should we consider aliased relations as CTE's?
        # Here we add it to our list of active cte's AFTER we visited the underlying relation without this alias
        # being present, so it's correct
        self.active_ctes[node.alias] = table
        if node.column_aliases:
            assert len(table.columns) == len(node.column_aliases)
            table = table.rename(
                {
                    orig: alias
                    for orig, alias in zip(table.columns, node.column_aliases)
                }
            )
        return table

    def visit_SelectQuery(self, node: SelectQuery) -> pl.LazyFrame:
        assert (
            node.from_ is not None
        ), "Empty FROM queries not yet supported. This is a constexpr table."
        assert node.window is None, "Window functions currently not supported."
        # The CTE's defined in this query will go out of scope afterwards, including overridden table names
        # TODO: For lateral queries we may need to preserve the active cte's
        with self.active_ctes, self.active_join_context:
            for relation in node.with_:
                # For a sequence of CTE's defined, one can use the previously defined CTE's
                # at any point:
                # trino> WITH f (a) AS (SELECT 1), g (b) AS (SELECT f.a FROM f) SELECT g.b FROM g;
                # b
                # ---
                # 1
                # (1 row)
                self.active_ctes[relation.alias] = self.visit(relation)
            table = self.visit(node.from_)

            if node.where:
                table = table.filter(self.expr_visitor.visit(node.where))

            projections = [self.expr_visitor.visit(val) for val in node.select]

            if node.groupby:
                groupby_nodes = self.get_groupby_exprs(node.groupby)
                # TODO: Add polars ticket for this - this is a whole mess due to groupbys including the
                #       grouped column
                # By default, SQL groupbys don't include the group that's being grouped on
                # so we should not include them by default and only include them if the user specifies them.
                group_columns = set(
                    get_expr_column_name(groupby) for groupby in groupby_nodes
                )
                projection_columns = [
                    get_expr_column_name(projection)
                    for projection in projections
                ]
                not_selected_group_cols = list(
                    group_columns - set(projection_columns)
                )
                projections = [
                    projection
                    for projection, col in zip(projections, projection_columns)
                    if col not in group_columns
                ]
                # We don't need to include the projected columns if it's already included by the groupby
                table = (
                    table.groupby(groupby_nodes)
                    .agg(projections)
                    .drop(not_selected_group_cols)
                )
            else:
                table = table.select(projections)

            if node.orderby:
                for orderby in node.orderby:
                    table = self.orderby_table(orderby, table)

            if node.limit:
                table = table.limit(node.limit)

            if node.select_quantifier == SetQuantifier.DISTINCT:
                table = table.unique()

            return table
