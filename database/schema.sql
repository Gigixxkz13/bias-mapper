CREATE TABLE IF NOT EXISTS PromptPair (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    bubble_type TEXT,
    topic TEXT,
    description TEXT,
    prompt_A_text TEXT,
    prompt_B_text TEXT
);

CREATE TABLE IF NOT EXISTS Run (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER,
    run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT,
    mode TEXT
);

CREATE TABLE IF NOT EXISTS Response (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    identifier TEXT,
    prompt_text TEXT,
    raw_output_text TEXT,
    sentimentScore REAL,
    diversityScore REAL,
    riskCount INTEGER,
    benefitCount INTEGER,
    emotionCount INTEGER
);

CREATE TABLE IF NOT EXISTS ListItem (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER,
    numberOfItem INTEGER,
    item_text TEXT,
    assigned_category TEXT
);