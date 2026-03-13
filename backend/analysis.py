import re


# -----------------------------
# Keyword dictionaries
# -----------------------------

benefit_keywords = {
    "opportunity", "growth", "success", "advantage", "benefit", "benefits"
}

risk_keywords = {
    "risk", "danger", "loss", "challenge", "problem", "problems"
}

emotion_keywords = {
    "emotional", "stressful", "exciting", "overwhelming", "frustrating", "anxious"
}


# -----------------------------
# Career categories for diversity
# -----------------------------

career_categories = {
    "Technology": {
        "developer",
        "programmer",
        "engineer",
        "software developer",
        "software engineer",
        "data scientist",
        "it specialist",
        "ai engineer",
        "machine learning"
    },

    "Healthcare": {
        "doctor",
        "physician",
        "nurse",
        "therapist",
        "pharmacist",
        "healthcare professional",
        "medical"
    },

    "Business": {
        "manager",
        "marketing specialist",
        "marketing manager",
        "financial analyst",
        "project coordinator",
        "consultant",
        "accountant",
        "entrepreneur",
        "business analyst"
    },

    "Education": {
        "teacher",
        "educator",
        "professor",
        "lecturer",
        "tutor"
    },

    "Arts": {
        "designer",
        "graphic designer",
        "artist",
        "writer",
        "photographer",
        "musician"
    },

    "Science": {
        "scientist",
        "environmental scientist",
        "researcher",
        "biologist",
        "chemist",
        "physicist"
    }
}


# -----------------------------
# Text preprocessing
# -----------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    return words


# -----------------------------
# Sentiment calculation
# -----------------------------

positive_words = {
    "good", "positive", "benefit", "benefits", "success", "growth", "advantage"
}

negative_words = {
    "bad", "negative", "risk", "danger", "loss", "problem", "stressful"
}


def calculate_sentiment(text):

    words = preprocess_text(text)

    positive_count = sum(1 for w in words if w in positive_words)
    negative_count = sum(1 for w in words if w in negative_words)

    total = positive_count + negative_count

    if total == 0:
        return 0.0

    score = (positive_count - negative_count) / total

    return round(score, 3)


# -----------------------------
# Keyword counting
# -----------------------------

def count_keywords(text):

    words = preprocess_text(text)

    risk_count = sum(1 for w in words if w in risk_keywords)
    benefit_count = sum(1 for w in words if w in benefit_keywords)
    emotion_count = sum(1 for w in words if w in emotion_keywords)

    return {
        "riskCount": risk_count,
        "benefitCount": benefit_count,
        "emotionCount": emotion_count
    }


# -----------------------------
# Diversity calculation
# -----------------------------

def categorize_item(item):

    item_lower = item.lower()

    for category, keywords in career_categories.items():
        for keyword in keywords:
            if keyword in item_lower:
                return category

    return "Other"


def calculate_diversity(items):

    if len(items) == 0:
        return None

    categories = set()

    for item in items:
        category = categorize_item(item)
        categories.add(category)

    diversity_score = len(categories) / len(items)

    return round(diversity_score, 3)