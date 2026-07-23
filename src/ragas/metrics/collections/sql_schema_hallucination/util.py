"""SQL schema hallucination utility functions."""

from __future__ import annotations

import typing as t

try:
    import sqlparse
    from sqlparse.sql import (
        Function,
        Identifier,
        IdentifierList,
        Parenthesis,
        TokenList,
        Where,
    )
    from sqlparse.tokens import DML, Keyword
except ImportError:  # pragma: no cover - surfaced at metric init with guidance
    sqlparse = None  # type: ignore[assignment]

Schema = t.Mapping[str, t.Sequence[str]]

# Keywords whose following identifiers name tables rather than columns.
_TABLE_CONTEXT = {"FROM", "INTO", "UPDATE"}
# Keywords that return us to a column context.
_COLUMN_CONTEXT = {"SELECT", "WHERE", "ON", "GROUP BY", "ORDER BY", "HAVING", "SET"}


class SQLReferences(t.NamedTuple):
    tables: t.Set[str]
    columns: t.Set[t.Tuple[t.Optional[str], str]]


class _Scope:
    """Table/column references collected within a single query scope.

    A scope is one ``SELECT`` (or DML statement). Subqueries and CTEs form their
    own nested scopes so their aliases never leak out as schema references.
    """

    def __init__(self) -> None:
        self.tables: t.Set[str] = set()
        self.alias_to_table: t.Dict[str, str] = {}
        self.virtual: t.Set[str] = set()  # CTE and derived-table names
        self.raw_columns: t.Set[t.Tuple[t.Optional[str], str]] = set()
        self.aliases: t.Set[str] = set()  # column aliases to ignore


def _clean(name: t.Optional[str]) -> t.Optional[str]:
    return name.strip('"`[]') if name else None


def _tokens(token: TokenList) -> t.Iterator[t.Any]:
    """Yield the meaningful children of a token, flattening an IdentifierList."""
    if isinstance(token, IdentifierList):
        yield from token.get_identifiers()  # type: ignore[attr-defined]
    else:
        yield token


def _has_select(token: TokenList) -> bool:
    return any(getattr(tok, "ttype", None) is DML for tok in token.flatten())


def _collect_ctes(statement: TokenList) -> t.Set[str]:
    ctes: t.Set[str] = set()
    after_with = False
    for token in statement.tokens:
        if token.is_whitespace:
            continue
        if token.normalized == "WITH":
            after_with = True
            continue
        if after_with:
            if isinstance(token, (Identifier, IdentifierList)):
                for child in _tokens(token):
                    name = _clean(child.get_real_name())
                    if name:
                        ctes.add(name)
            after_with = False
    return ctes


def _walk(statement: TokenList, context: str, scope: _Scope) -> None:
    for token in statement.tokens:
        if token.is_whitespace:
            continue

        if token.ttype is DML:
            context = "TABLE" if token.normalized.upper() == "UPDATE" else "COLUMN"
            continue
        if token.ttype is Keyword:
            upper = token.normalized.upper()
            if upper in _TABLE_CONTEXT or upper.endswith("JOIN"):
                context = "TABLE"
            elif upper in _COLUMN_CONTEXT:
                context = "COLUMN"
            continue

        if isinstance(token, Where):
            _walk(token, "COLUMN", scope)
            continue
        if isinstance(token, Parenthesis):
            _walk(token, "COLUMN", scope)
            continue

        if isinstance(token, (Identifier, IdentifierList)):
            for child in _tokens(token):
                _handle(child, context, scope)
            continue

        if isinstance(token, Function):
            _handle(token, context, scope)
            continue

        # Recurse into grouped tokens like Comparison or Operation so nested
        # identifiers (e.g. a WHERE predicate) are still classified.
        if isinstance(token, TokenList):
            _walk(token, context, scope)


def _handle(identifier: t.Any, context: str, scope: _Scope) -> None:
    # A function call: its name is not a column, but its arguments may be. In a
    # table context ``table (col, col)`` also parses as a function (e.g. the
    # target of INSERT INTO), so the name is the table there.
    if isinstance(identifier, Function):
        name = _clean(identifier.get_real_name())
        if context == "TABLE" and name:
            scope.tables.add(name)
        paren = next(
            (tok for tok in identifier.tokens if isinstance(tok, Parenthesis)), None
        )
        if paren is not None:
            _walk(paren, "COLUMN", scope)
        return
    if not isinstance(identifier, Identifier):
        return

    paren = next(
        (tok for tok in identifier.tokens if isinstance(tok, Parenthesis)), None
    )
    if paren is not None:
        alias = _clean(identifier.get_alias())
        if _has_select(paren):
            # Derived table: (SELECT ...) alias. The alias is virtual and its
            # inner scope is walked so real tables inside are still checked.
            if alias:
                scope.virtual.add(alias)
                scope.aliases.add(alias)
            _walk(paren, "COLUMN", scope)
            return
        # INSERT-style ``table (col, col)``: the name is a table, parens columns.
        name = _clean(identifier.get_real_name())
        if name and context == "TABLE":
            scope.tables.add(name)
        _walk(paren, "COLUMN", scope)
        return

    alias = _clean(identifier.get_alias())
    if alias:
        scope.aliases.add(alias)

    real = _clean(identifier.get_real_name())
    if not real:
        return

    if context == "TABLE":
        scope.tables.add(real)
        if alias:
            scope.alias_to_table[alias] = real
    else:
        qualifier = _clean(identifier.get_parent_name())
        scope.raw_columns.add((qualifier, real))


def _resolve(scope: _Scope) -> SQLReferences:
    tables = scope.tables - scope.virtual

    columns: t.Set[t.Tuple[t.Optional[str], str]] = set()
    for qualifier, name in scope.raw_columns:
        if name in scope.aliases:
            continue
        if qualifier is None:
            columns.add((None, name))
            continue
        if qualifier in scope.virtual:
            continue
        resolved = scope.alias_to_table.get(qualifier, qualifier)
        if resolved in scope.virtual:
            continue
        columns.add((resolved, name))

    return SQLReferences(tables=tables, columns=columns)


def extract_references(sql: str) -> SQLReferences:
    """Extract table and column references from a SQL query.

    Handles multiple statements, joins, functions, subqueries, CTEs and simple
    DML. In-query aliases and CTE / derived-table names are resolved and never
    counted as references. ``*`` wildcards and function names are ignored.

    Note: schema-qualified names deeper than ``table.column`` (e.g.
    ``db.schema.table``) are not resolved, matching the flat ``{table: columns}``
    schema contract.
    """
    assert sqlparse is not None, "sqlparse is required for SQLSchemaHallucination"

    tables: t.Set[str] = set()
    columns: t.Set[t.Tuple[t.Optional[str], str]] = set()
    for statement in sqlparse.parse(sql):
        scope = _Scope()
        scope.virtual |= _collect_ctes(statement)
        _walk(statement, "COLUMN", scope)
        refs = _resolve(scope)
        tables |= refs.tables
        columns |= refs.columns

    return SQLReferences(tables=tables, columns=columns)


def _normalize_schema(schema: Schema) -> t.Dict[str, t.Set[str]]:
    return {
        table.lower(): {col.lower() for col in cols} for table, cols in schema.items()
    }


def score_references(
    references: SQLReferences,
    schema: Schema,
) -> t.Tuple[int, int, t.List[str]]:
    """Count references that exist in the schema.

    Unqualified columns are validated against the columns of the tables the
    query actually references, not the whole schema, so a column that exists in
    some unrelated table is still flagged. Returns (valid, total, hallucinated).
    """
    normalized = _normalize_schema(schema)

    referenced_columns: t.Set[str] = set()
    for table in references.tables:
        referenced_columns |= normalized.get(table.lower(), set())
    if not referenced_columns:
        # No referenced table is in the schema; fall back to the whole schema so
        # a bad table name does not also flag every column as hallucinated.
        referenced_columns = {col for cols in normalized.values() for col in cols}

    valid = 0
    total = 0
    hallucinated: t.List[str] = []

    for table in references.tables:
        total += 1
        if table.lower() in normalized:
            valid += 1
        else:
            hallucinated.append(table)

    for qualifier, column in references.columns:
        total += 1
        col = column.lower()
        if qualifier is not None:
            table = qualifier.lower()
            if table in normalized and col in normalized[table]:
                valid += 1
            else:
                hallucinated.append(f"{qualifier}.{column}")
        elif col in referenced_columns:
            valid += 1
        else:
            hallucinated.append(column)

    return valid, total, hallucinated
