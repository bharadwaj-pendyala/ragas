"""Tests for SQLSchemaHallucination metric (collections implementation)."""

import pytest

from ragas.metrics.collections import SQLSchemaHallucination
from ragas.metrics.collections.sql_schema_hallucination.util import (
    extract_references,
    score_references,
)

SCHEMA = {
    "users": ["id", "name", "email"],
    "orders": ["id", "user_id", "total"],
}


class TestSQLReferenceExtraction:
    """Test cases for the SQL parsing utilities."""

    def test_extracts_table_and_columns(self):
        refs = extract_references("SELECT id, name FROM users")
        assert refs.tables == {"users"}
        assert refs.columns == {(None, "id"), (None, "name")}

    def test_wildcard_is_ignored(self):
        refs = extract_references("SELECT * FROM users")
        assert refs.columns == set()
        assert refs.tables == {"users"}

    def test_function_name_is_not_a_column(self):
        refs = extract_references("SELECT COUNT(*) FROM users")
        assert refs.columns == set()

    def test_column_alias_is_excluded(self):
        refs = extract_references("SELECT id AS user_id FROM users")
        assert refs.columns == {(None, "id")}

    def test_table_alias_resolves_qualified_column(self):
        refs = extract_references("SELECT u.email FROM users u")
        assert refs.columns == {("users", "email")}

    def test_cte_name_is_not_a_table(self):
        refs = extract_references("WITH t AS (SELECT id FROM users) SELECT id FROM t")
        assert "t" not in refs.tables
        assert "users" in refs.tables

    def test_where_clause_columns_are_extracted(self):
        refs = extract_references("SELECT name FROM users WHERE emial = 'x'")
        assert (None, "emial") in refs.columns

    def test_empty_query_yields_no_references(self):
        refs = extract_references("")
        assert refs.tables == set()
        assert refs.columns == set()

    def test_score_counts_valid_and_hallucinated(self):
        refs = extract_references("SELECT emial FROM users")
        valid, total, hallucinated = score_references(refs, SCHEMA)
        assert (valid, total) == (1, 2)
        assert hallucinated == ["emial"]


class TestSQLSchemaHallucination:
    """Test cases for the SQLSchemaHallucination metric."""

    def test_init_default_name(self):
        metric = SQLSchemaHallucination()
        assert metric.name == "sql_schema_hallucination"

    def test_init_custom_name(self):
        metric = SQLSchemaHallucination(name="custom")
        assert metric.name == "custom"

    @pytest.mark.asyncio
    async def test_all_references_valid(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT id, name FROM users", schema=SCHEMA
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_hallucinated_column(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(response="SELECT emial FROM users", schema=SCHEMA)
        assert result.value == 0.5
        assert "emial" in result.reason

    @pytest.mark.asyncio
    async def test_hallucinated_table(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(response="SELECT id FROM userz", schema=SCHEMA)
        assert result.value == 0.5
        assert "userz" in result.reason

    @pytest.mark.asyncio
    async def test_qualified_column_via_alias(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT o.total FROM orders o JOIN users u ON o.user_id = u.id",
            schema=SCHEMA,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_select_star_scores_one(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(response="SELECT * FROM users", schema=SCHEMA)
        assert result.value == 1.0
        assert result.reason == "Valid references 1/1"

    @pytest.mark.asyncio
    async def test_cte_name_not_flagged(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="WITH recent AS (SELECT id FROM users) SELECT id FROM recent",
            schema=SCHEMA,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT ID, NAME FROM USERS", schema=SCHEMA
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_empty_response_raises(self):
        metric = SQLSchemaHallucination()
        with pytest.raises(ValueError, match="non-empty SQL query"):
            await metric.ascore(response="  ", schema=SCHEMA)

    @pytest.mark.asyncio
    async def test_empty_schema_raises(self):
        metric = SQLSchemaHallucination()
        with pytest.raises(ValueError, match="non-empty mapping"):
            await metric.ascore(response="SELECT id FROM users", schema={})

    def test_sync_score_method(self):
        metric = SQLSchemaHallucination()
        result = metric.score(response="SELECT emial FROM users", schema=SCHEMA)
        assert result.value == 0.5
