import re

def extract_sql(llm_output: str) -> str:
    """Extract clean SQL from LLM response that may contain explanations."""
    # Case 1: SQL inside ```sql ... ``` blocks
    sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', llm_output, re.DOTALL)
    if sql_blocks:
        return sql_blocks[0].strip()

    # Case 2: SQL inside generic ``` ... ``` blocks
    code_blocks = re.findall(r'```\s*(.*?)\s*```', llm_output, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Case 3: Response starts with a SQL keyword (already clean)
    cleaned = llm_output.strip()
    if cleaned.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
        return cleaned

    # Case 4: Find a SELECT statement somewhere in the text
    match = re.search(r'(SELECT\s+.*?;)', llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Case 5: Find SELECT without semicolon
    match = re.search(r'(SELECT\s+.+)', llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return cleaned  # Last resort


def clean_sql(sql: str) -> str:
    """Fix common LLM SQL quoting mistakes for PostgreSQL."""
    # Fix "table.column" → table.column (remove quotes around dotted names)
    sql = re.sub(r'"(\w+)\.(\w+)"', r'\1.\2', sql)
    # Fix "table"."column" → table.column (remove unnecessary quotes)
    sql = re.sub(r'"(\w+)"\."(\w+)"', r'\1.\2', sql)
    # Fix "alias" → alias for simple table aliases
    sql = re.sub(r'"(\w{1,3})"\.', r'\1.', sql)
    # Remove trailing semicolons
    sql = sql.rstrip(';').strip()
    return sql
