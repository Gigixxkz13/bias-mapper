import sqlite3

DB_PATH = "database/biasmapper.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()

    with open("database/schema.sql", "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()


def insert_prompt_pair(name, bubble_type, topic, description, prompt_A_text, prompt_B_text):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO PromptPair (name, bubble_type, topic, description, prompt_A_text, prompt_B_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, bubble_type, topic, description, prompt_A_text, prompt_B_text))

    conn.commit()
    pair_id = cursor.lastrowid
    conn.close()

    return pair_id


def create_run(pair_id, model_name, mode):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO Run (pair_id, model_name, mode)
        VALUES (?, ?, ?)
    """, (pair_id, model_name, mode))

    conn.commit()
    run_id = cursor.lastrowid
    conn.close()

    return run_id


def insert_response(run_id, identifier, prompt_text, raw_output_text,
                    sentimentScore=None, diversityScore=None,
                    riskCount=None, benefitCount=None, emotionCount=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO Response (
            run_id, identifier, prompt_text, raw_output_text,
            sentimentScore, diversityScore, riskCount, benefitCount, emotionCount
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, identifier, prompt_text, raw_output_text,
        sentimentScore, diversityScore, riskCount, benefitCount, emotionCount
    ))

    conn.commit()
    response_id = cursor.lastrowid
    conn.close()

    return response_id


def insert_list_item(response_id, numberOfItem, item_text, assigned_category):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO ListItem (response_id, numberOfItem, item_text, assigned_category)
        VALUES (?, ?, ?, ?)
    """, (response_id, numberOfItem, item_text, assigned_category))

    conn.commit()
    item_id = cursor.lastrowid
    conn.close()

    return item_id

def get_all_runs():
    conn = get_connection()
    cursor = conn.cursor()

    # join with PromptPair so each run row includes name and bubble_type
    # these are needed for the history page table display
    # _______________________________________________________
    cursor.execute("""
        SELECT
            r.run_id,
            r.pair_id,
            r.run_timestamp,
            r.model_name,
            r.mode,
            p.name        AS name,
            p.bubble_type AS bubble_type
        FROM Run r
        LEFT JOIN PromptPair p ON r.pair_id = p.pair_id
        ORDER BY r.run_id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_run_by_id(run_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT run_id, pair_id, run_timestamp, model_name, mode
        FROM Run
        WHERE run_id = ?
    """, (run_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def get_prompt_pair_by_id(pair_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pair_id, name, bubble_type, topic, description, prompt_A_text, prompt_B_text
        FROM PromptPair
        WHERE pair_id = ?
    """, (pair_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def get_responses_by_run_id(run_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT response_id, run_id, identifier, prompt_text, raw_output_text,
               sentimentScore, diversityScore, riskCount, benefitCount, emotionCount
        FROM Response
        WHERE run_id = ?
        ORDER BY identifier
    """, (run_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_list_items_by_response_id(response_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT item_id, response_id, numberOfItem, item_text, assigned_category
        FROM ListItem
        WHERE response_id = ?
        ORDER BY numberOfItem
    """, (response_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]