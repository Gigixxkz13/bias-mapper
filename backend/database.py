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