import polars as pl
import treeno.datatypes.types as tt
from treeno.base import PrintOptions

_TIME_PRECISION_MAP = {
    0: "s",
    3: "ms",
    6: "us",
    9: "ns",
}


def _time_precision_to_unit(precision: int) -> str:
    if precision not in _TIME_PRECISION_MAP:
        raise NotImplementedError(
            f"Precision {precision} does not have a perfect arrow unit mapping"
        )
    return _TIME_PRECISION_MAP(precision)


def to_polars_datatype(data_type: tt.DataType) -> pl.DataType:
    # TODO: Let's replace this with structural pattern matching when
    # we bump to python 3.10+!
    if data_type.type_name == tt.BOOLEAN:
        return pl.Boolean
    if data_type.type_name == tt.INTEGER:
        return pl.Int32
    if data_type.type_name == tt.TINYINT:
        return pl.Int8
    if data_type.type_name == tt.SMALLINT:
        return pl.Int16
    if data_type.type_name == tt.BIGINT:
        return pl.Int64
    if data_type.type_name == tt.REAL:
        return pl.Float32
    if data_type.type_name == tt.DOUBLE:
        return pl.Float64
    if data_type.type_name == tt.DECIMAL:
        raise NotImplementedError("Decimal types are not supported in Polars")
    # VARCHAR and friends can have max_chars limit, but arrow doesn't
    if data_type.type_name in (tt.VARCHAR, tt.CHAR):
        return pl.Utf8
    # Trino doesn't allow max length, but arrow does
    if data_type.type_name == tt.VARBINARY:
        raise NotImplementedError("Binary types are not supported in Polars")
    if data_type.type_name == tt.TIME:
        assert (
            "precision" in data_type.parameters
        ), f"Time type must have precision defined, got {data_type.sql(PrintOptions())}"
        assert (
            "timezone" not in data_type.parameters
        ), f"Arrow doesn't have a tz-aware time type, got {data_type.sql(PrintOptions())}"
        unit = _time_precision_to_unit(data_type.parameters["precision"])
        return pl.Duration(unit)
    if data_type.type_name == tt.TIMESTAMP:
        assert (
            "precision" in data_type.parameters
        ), f"Timestamp type must have precision defined, got {data_type.sql(PrintOptions())}"
        assert "timezone" not in data_type.parameters, (
            "Polar's Datetime type requires a fixed timezone(e.g. 'America/New_York'), but "
            "Trino's timestamp type can have dynamic timezone information, "
            f"got {data_type.sql(PrintOptions())}"
        )
        return pl.Datetime(
            _time_precision_to_unit(data_type.parameters["precision"])
        )
    raise NotImplementedError(
        f"Data type conversion for {data_type.sql(PrintOptions())} to polars is not supported"
    )
