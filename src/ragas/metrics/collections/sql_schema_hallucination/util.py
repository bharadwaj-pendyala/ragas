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
_TABLE_CONTEXT = {"FROM", "JOIN", "INTO", "UPDATE"}
# Keywords that return us to a column context.
_COLUMN_CONTEXT = {"SELECT", "WHERE", "ON", "GROUP BY", "ORDER BY", "HAVING", "SET"}


class SQLReferences(t.NamedTuple):
    tables: t.Set[str]
    columns: t.Set[t.Tuple[t.Optional[str], str]]


def _clean(name: t.Optional[str]) -> t.Optional[str]:
    return name.strip('"`[]') if name else None


def _children(token: TokenList) -> t.Iterator[Identifier]:
    """Yield Identifier children, flattening an IdentifierList."""
    if isinstance(token, IdentifierList):
        source: t.Iterable[t.Any] = token.get_identifiers()  # type: ignore[attr-defined]
    else:
        source = [token]
    for child in source:
        if isinstance(child, Identifier):
            yield child


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
                for identifier in _children(token):
                    name = _clean(identifier.get_real_name())
                    if name:
                        ctes.add(name)
            after_with = False
    return ctes


def _walk(
    statement: TokenList,
    context: str,
    tables: t.Set[str],
    alias_to_table: t.Dict[str, str],
    columns: t.Set[t.Tuple[t.Optional[str], str]],
    aliases: t.Set[str],
) -> None:
    for token in statement.tokens:
        if token.is_whitespace:
            continue

        if token.ttype is DML and token.normalized == "SELECT":
            context = "COLUMN"
            continue
        if token.ttype is Keyword:
            upper = token.normalized.upper()
            if upper in _TABLE_CONTEXT or upper.endswith("JOIN"):
                context = "TABLE"
            elif upper in _COLUMN_CONTEXT:
                context = "COLUMN"
            continue

        if isinstance(token, Where):
            _walk(token, "COLUMN", tables, alias_to_table, columns, aliases)
            continue
        if isinstance(token, Parenthesis):
            _walk(token, "COLUMN", tables, alias_to_table, columns, aliases)
            continue

        if isinstance(token, (Identifier, IdentifierList)):
            for identifier in _children(token):
                _handle_identifier(
                    identifier, context, tables, alias_to_table, columns, aliases
                )
            continue

        # Recurse into grouped tokens like Comparison or Operation (e.g. a WHERE
        # predicate) so identifiers nested inside them are still classified.
        if isinstance(token, TokenList) and not isinstance(token, Function):
            _walk(token, context, tables, alias_to_table, columns, aliases)


def _handle_identifier(
    identifier: Identifier,
    context: str,
    tables: t.Set[str],
    alias_to_table: t.Dict[str, str],
    columns: t.Set[t.Tuple[t.Optional[str], str]],
    aliases: t.Set[str],
) -> None:
    if isinstance(identifier, Function):
        return

    # A subquery aliased as a derived table: recurse, don't treat as a reference.
    if any(isinstance(tok, Parenthesis) for tok in identifier.tokens):
        _walk(identifier, "COLUMN", tables, alias_to_table, columns, aliases)
        return

    alias = _clean(identifier.get_alias())
    if alias:
        aliases.add(alias)

    real = _clean(identifier.get_real_name())
    if not real:
        return

    if context == "TABLE":
        tables.add(real)
        if alias:
            alias_to_table[alias] = real
    else:
        qualifier = _clean(identifier.get_parent_name())
        columns.add((qualifier, real))


def extract_references(sql: str) -> SQLReferences:
    """Extract table and column references from a SQL query.

    In-query aliases and CTE names are excluded, as are ``*`` wildcards and
    function calls. Qualifiers on columns are resolved through table aliases so
    ``u.email`` on ``FROM users u`` validates against ``users``.
    """
    assert sqlparse is not None, "sqlparse is required for SQLSchemaHallucination"
    parsed = sqlparse.parse(sql)
    if not parsed:
        return SQLReferences(tables=set(), columns=set())
    statement = parsed[0]
    ctes = _collect_ctes(statement)

    tables: t.Set[str] = set()
    alias_to_table: t.Dict[str, str] = {}
    raw_columns: t.Set[t.Tuple[t.Optional[str], str]] = set()
    aliases: t.Set[str] = set()
    _walk(statement, "COLUMN", tables, alias_to_table, raw_columns, aliases)

    tables -= ctes

    columns: t.Set[t.Tuple[t.Optional[str], str]] = set()
    for qualifier, name in raw_columns:
        if name in aliases:
            continue
        resolved = alias_to_table.get(qualifier, qualifier) if qualifier else None
        columns.add((resolved, name))

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

    Returns (valid_count, total_count, hallucinated) where ``hallucinated`` lists
    the offending references for diagnostics.
    """
    normalized = _normalize_schema(schema)
    all_columns = {col for cols in normalized.values() for col in cols}

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
        elif col in all_columns:
            valid += 1
        else:
            hallucinated.append(column)

    return valid, total, hallucinated
