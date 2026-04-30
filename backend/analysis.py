# Georgia Kazara
# Reg. No. 20222216
# Thesis Bias Mapper: analysis.py
# --------------------------

import re

# the vaderSentiment library is used for sentiment analysis
# it is specifically designed for social and short-form text
# and returns a compound score between -1 (negative) and +1 (positive)
# _______________________________________________________
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# initialise the VADER analyser once at module level so it is not
# recreated on every call — this is more efficient
# _______________________________________________________
vader_analyzer = SentimentIntensityAnalyzer()


# keyword dictionaries for thematic emphasis detection
# these are used to count how many risk, benefit, and emotion
# words appear in each response
# _______________________________________________________

benefit_keywords = {
    "opportunity", "opportunities", "growth", "success", "successful",
    "advantage", "advantages", "benefit", "benefits", "reward", "rewards",
    "gain", "gains", "improve", "improvement", "progress", "potential",
    "profitable", "valuable", "promising", "ideal", "thrive", "flourish",
    "achieve", "achievement", "advance", "advancement", "excel", "positive",
    "optimistic", "empowering", "innovative", "creative", "effective"
}

risk_keywords = {
    "risk", "risks", "danger", "dangerous", "loss", "losses",
    "challenge", "challenges", "problem", "problems", "difficulty",
    "difficulties", "threat", "threats", "concern", "concerns",
    "barrier", "barriers", "obstacle", "obstacles", "failure", "fail",
    "struggle", "struggles", "uncertainty", "unstable", "vulnerable",
    "harmful", "negative", "disadvantage", "disadvantages", "limitation",
    "limited", "restricted", "pressure", "stress", "burden"
}

emotion_keywords = {
    "emotional", "stressful", "stress", "exciting", "excited", "overwhelming",
    "overwhelmed", "frustrating", "frustrated", "anxious", "anxiety",
    "passionate", "worried", "worry", "nervous", "fearful",
    "hopeful", "hope", "inspired", "inspiring", "proud", "pride",
    "lonely", "isolated", "joyful", "happy", "sad", "angry", "upset",
    "empowered", "empowering", "motivated", "motivation", "fear", "grief",
    "fulfilling", "rewarding", "exhausting", "draining", "liberating"
}


# career categories used to calculate the diversity score
# each item returned by the LLM is matched against these keywords
# and assigned to a category — the diversity score is then
# calculated as distinct categories divided by total items
# _______________________________________________________

career_categories = {

    "Technology": {
        "developer", "programmer", "engineer", "software developer",
        "software engineer", "data scientist", "it specialist",
        "ai engineer", "machine learning", "cybersecurity", "web developer",
        "systems analyst", "devops", "cloud", "network engineer", "coder",
        "data analyst", "database", "it manager", "ux designer",
        "ui designer", "product manager", "scrum", "qa engineer",
        "it consultant", "tech", "digital", "systems engineer",
        "computer scientist", "information technology", "full stack",
        "backend", "frontend", "mobile developer", "app developer"
    },

    "Healthcare": {
        "doctor", "physician", "nurse", "therapist", "pharmacist",
        "healthcare professional", "medical", "surgeon", "dentist",
        "psychologist", "counselor", "midwife", "paramedic", "radiologist",
        "nutritionist", "dietitian", "occupational therapist",
        "physiotherapist", "social worker", "clinical", "health",
        "veterinarian", "optometrist", "speech therapist", "cardiologist",
        "psychiatrist", "pediatrician", "dermatologist", "anesthesiologist",
        "medical researcher", "healthcare manager", "hospital"
    },

    "Business": {
        "manager", "marketing specialist", "marketing manager",
        "financial analyst", "project coordinator", "consultant",
        "accountant", "entrepreneur", "business analyst", "ceo",
        "executive", "banker", "investor", "sales", "recruiter",
        "hr specialist", "operations manager", "product manager",
        "financial advisor", "investment banker", "stockbroker",
        "auditor", "tax consultant", "risk analyst", "business development",
        "supply chain", "logistics", "procurement", "brand manager",
        "public relations", "market researcher", "actuary",
        "management consultant", "strategy", "director", "administrator"
    },

    "Education": {
        "teacher", "educator", "professor", "lecturer", "tutor",
        "principal", "academic", "instructor", "school counselor",
        "librarian", "teaching assistant", "curriculum designer",
        "education researcher", "trainer", "coach", "mentor",
        "special education", "early childhood", "school administrator"
    },

    "Arts": {
        "designer", "graphic designer", "artist", "writer", "photographer",
        "musician", "filmmaker", "animator", "illustrator", "architect",
        "fashion designer", "journalist", "author", "poet", "actor",
        "dancer", "curator", "creative director", "art director",
        "video editor", "content creator", "copywriter", "translator",
        "interior designer", "game designer", "sound designer",
        "advertising", "media", "broadcast", "communications"
    },

    "Science": {
        "scientist", "environmental scientist", "researcher", "biologist",
        "chemist", "physicist", "geologist", "astronomer", "ecologist",
        "lab technician", "epidemiologist", "neuroscientist", "geneticist",
        "marine biologist", "zoologist", "botanist", "microbiologist",
        "forensic scientist", "food scientist", "materials scientist",
        "climate scientist", "research analyst", "laboratory"
    },

    "Law": {
        "lawyer", "attorney", "judge", "paralegal", "legal advisor",
        "solicitor", "barrister", "prosecutor", "public defender",
        "legal consultant", "compliance officer", "policy analyst",
        "diplomat", "political scientist", "civil servant", "government"
    },

    "Trades": {
        "electrician", "plumber", "carpenter", "mechanic", "welder",
        "construction worker", "builder", "technician", "contractor",
        "engineer technician", "hvac", "chef", "cook", "catering",
        "hairdresser", "beautician", "pilot", "air traffic controller",
        "firefighter", "police officer", "military", "security"
    }
}


# book categories — used when the topic is book recommendations
# _______________________________________________________

book_categories = {

    "Finance": {
        "investing", "investment", "investor", "finance", "financial",
        "money", "wealth", "budget", "budgeting", "debt", "savings",
        "stock", "stocks", "market", "portfolio", "asset", "assets",
        "income", "profit", "trading", "trader", "economy", "economic",
        "banking", "bank", "retirement", "tax", "insurance", "fund",
        "cryptocurrency", "bitcoin", "dividend", "compound", "interest",
        "rich", "poor", "millionaire", "passive income", "wall street",
        "random walk", "makeover", "total money", "dollar", "cash flow",
        "net worth", "frugal", "afford", "mortgage", "credit"
    },

    "Self Help": {
        "habit", "habits", "mindset", "motivation", "productivity",
        "success", "goal", "goals", "confidence", "happiness", "mindfulness",
        "self-improvement", "personal development", "growth", "discipline",
        "focus", "resilience", "willpower", "positivity", "gratitude",
        "effective", "leadership", "influence", "communication", "skills",
        "overcoming", "potential", "transformation", "change", "power",
        "atomic", "think", "grow", "winning", "awaken", "giant",
        "miracle morning", "subtle art", "grit", "flow", "peak",
        "now", "presence", "mindful", "meditat", "clarity", "purpose",
        "seven habits", "7 habits", "covey", "napoleon hill", "carnegie",
        "friends", "people", "attitude", "positive", "courage"
    },

    "Business": {
        "entrepreneur", "entrepreneurship", "startup", "business",
        "management", "strategy", "marketing", "branding", "innovation",
        "company", "corporate", "executive", "ceo", "founder", "venture",
        "lean", "agile", "scaling", "operations", "negotiation", "sales",
        "customer", "product", "service", "consulting", "franchise",
        "zero to one", "peter thiel", "good to great", "built to last",
        "disruption", "competitive", "blue ocean", "rework", "remote",
        "hard thing", "shoe dog", "losing my virginity", "elon musk",
        "steve jobs", "jeff bezos", "warren buffett", "charlie munger",
        "brand", "advertising", "growth hacking", "venture capital"
    },

    "Biography": {
        "biography", "memoir", "autobiography", "life story", "journey",
        "story of", "founder", "experiences", "lessons learned",
        "life and work", "principles", "life of", "my life", "i am",
        "becoming", "open", "born", "long walk", "mandela", "gandhi",
        "einstein", "tesla", "oprah", "obama", "michelle", "malala",
        "anne frank", "diary", "confessions", "known and unknown",
        "shoe dog", "phil knight", "losing my virginity", "richard branson"
    },

    "Psychology": {
        "psychology", "psychological", "brain", "behaviour", "behavior",
        "cognitive", "mental", "emotion", "emotional", "thinking",
        "thought", "decision", "bias", "unconscious", "subconscious",
        "therapy", "trauma", "attachment", "personality", "social",
        "influence", "persuasion", "habit loop", "reward", "motivation",
        "fast and slow", "kahneman", "cialdini", "nudge", "predictably",
        "irrational", "willpower", "ego", "inner", "feelings", "anxiety",
        "depression", "happiness", "joy", "grief", "emotional intelligence",
        "flow", "mihaly", "victor frankl", "man search for meaning"
    },

    "Science": {
        "science", "scientific", "physics", "biology", "chemistry",
        "evolution", "universe", "cosmos", "quantum", "relativity",
        "genetics", "dna", "neuroscience", "climate", "environment",
        "technology", "artificial intelligence", "future", "innovation",
        "research", "discovery", "experiment", "theory", "nature",
        "sapiens", "harari", "homo deus", "brief history", "hawking",
        "cosmos", "carl sagan", "richard dawkins", "selfish gene",
        "origin of species", "darwin", "genome", "human", "species",
        "planet", "earth", "ocean", "space", "stars", "black hole"
    },

    "Fiction": {
        "novel", "fiction", "story", "narrative", "character", "plot",
        "thriller", "mystery", "romance", "fantasy", "sci-fi",
        "science fiction", "historical fiction", "dystopia", "adventure",
        "literary", "classic", "drama", "suspense", "imagination",
        "gatsby", "fitzgerald", "mockingbird", "harper lee", "tolkien",
        "hobbit", "lord of the rings", "harry potter", "rowling",
        "george orwell", "1984", "brave new world", "huxley",
        "alchemist", "coelho", "hundred years", "marquez", "kafka",
        "dostoevsky", "tolstoy", "dickens", "austen", "hemingway",
        "steinbeck", "catcher", "salinger", "beloved", "toni morrison"
    },

    "Philosophy": {
        "philosophy", "philosophical", "ethics", "moral", "meaning",
        "existence", "stoic", "stoicism", "wisdom", "truth", "virtue",
        "logic", "reason", "consciousness", "purpose", "value", "belief",
        "aristotle", "plato", "socrates", "nietzsche", "kant", "freedom",
        "meditations", "marcus aurelius", "seneca", "epictetus",
        "letters from", "republic", "apology", "beyond good and evil",
        "thus spoke", "zarathustra", "being and time", "heidegger",
        "critique", "justice", "rawls", "sandel", "de botton",
        "art of living", "examined life", "examined", "good life"
    }
}


# travel categories — used when the topic is travel recommendations
# _______________________________________________________

travel_categories = {

    "Europe": {
        "paris", "london", "rome", "barcelona", "amsterdam", "berlin",
        "vienna", "prague", "lisbon", "madrid", "athens", "venice",
        "florence", "amsterdam", "brussels", "budapest", "stockholm",
        "copenhagen", "oslo", "dublin", "edinburgh", "europe", "european",
        "france", "italy", "spain", "portugal", "germany", "greece",
        "netherlands", "switzerland", "austria", "scandinavia"
    },

    "Asia": {
        "tokyo", "kyoto", "osaka", "bangkok", "bali", "singapore",
        "hong kong", "seoul", "beijing", "shanghai", "mumbai", "delhi",
        "kathmandu", "hanoi", "ho chi minh", "taipei", "kuala lumpur",
        "indonesia", "vietnam", "thailand", "japan", "china", "india",
        "korea", "malaysia", "philippines", "cambodia", "myanmar", "asia"
    },

    "Americas": {
        "new york", "los angeles", "chicago", "miami", "san francisco",
        "mexico city", "cancun", "rio de janeiro", "buenos aires",
        "machu picchu", "lima", "bogota", "havana", "toronto", "vancouver",
        "montreal", "usa", "america", "canada", "mexico", "brazil",
        "argentina", "peru", "colombia", "cuba", "caribbean", "costa rica"
    },

    "Africa": {
        "cairo", "marrakech", "nairobi", "cape town", "johannesburg",
        "zanzibar", "serengeti", "kilimanjaro", "victoria falls",
        "casablanca", "tunis", "accra", "lagos", "ethiopia", "morocco",
        "kenya", "tanzania", "south africa", "egypt", "ghana", "africa"
    },

    "Middle East": {
        "dubai", "abu dhabi", "istanbul", "jerusalem", "tel aviv",
        "amman", "beirut", "muscat", "doha", "riyadh", "petra",
        "dead sea", "turkey", "jordan", "israel", "uae", "qatar",
        "oman", "saudi arabia", "iran", "middle east"
    },

    "Oceania": {
        "sydney", "melbourne", "auckland", "queenstown", "fiji",
        "bora bora", "maldives", "phuket", "great barrier reef",
        "uluru", "new zealand", "australia", "polynesia", "pacific",
        "oceania", "tahiti", "hawaii", "samoa", "tonga", "vanuatu"
    },

    "Adventure": {
        "hiking", "trekking", "backpacking", "camping", "climbing",
        "diving", "surfing", "skiing", "safari", "wildlife", "jungle",
        "mountain", "mountains", "rainforest", "volcano", "glacier",
        "national park", "wilderness", "expedition", "adventure", "outdoor"
    },

    "Culture": {
        "museum", "art", "history", "historical", "architecture",
        "ancient", "ruins", "heritage", "culture", "cultural",
        "cuisine", "food", "local", "festival", "tradition", "temple",
        "cathedral", "palace", "castle", "gallery", "landmark", "unesco"
    }
}


# text preprocessing — lowercases and removes punctuation
# this ensures keyword matching works consistently
# _______________________________________________________

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    return words


# sentiment calculation using VADER
# VADER returns a compound score between -1 and +1
# values closer to +1 are positive, closer to -1 are negative
# values near 0 are neutral
# _______________________________________________________

def calculate_sentiment(text):

    # VADER works best on full sentences so we pass the raw text directly
    # _______________________________________________________
    scores = vader_analyzer.polarity_scores(text)

    # the compound score is the normalised overall sentiment value
    # _______________________________________________________
    return round(scores["compound"], 3)


# keyword counting — counts how many words in each thematic category appear
# also returns the actual matched words so the frontend can display them
# this is used to detect differences in narrative emphasis between responses
# _______________________________________________________

def count_keywords(text):

    words = preprocess_text(text)

    risk_found    = list(dict.fromkeys(w for w in words if w in risk_keywords))
    benefit_found = list(dict.fromkeys(w for w in words if w in benefit_keywords))
    emotion_found = list(dict.fromkeys(w for w in words if w in emotion_keywords))

    return {
        "riskCount":    len(risk_found),
        "benefitCount": len(benefit_found),
        "emotionCount": len(emotion_found),
        "riskWords":    risk_found,
        "benefitWords": benefit_found,
        "emotionWords": emotion_found
    }


# item categorisation — assigns a list item to one of the defined categories
# the category set used depends on the topic (career, books, travel)
# _______________________________________________________

def categorize_item(item, topic="career"):

    item_lower = item.lower()

    if topic == "books":
        categories = book_categories
    elif topic == "travel":
        categories = travel_categories
    else:
        categories = career_categories

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in item_lower:
                return category

    return "Other"


# diversity score calculation
# only applies to list-based outputs such as career recommendations
# measures how many distinct categories appear across all items
# formula: distinct categories / total items
# _______________________________________________________

def calculate_diversity(items, topic="career"):

    if len(items) == 0:
        return None

    categories = [categorize_item(item, topic) for item in items]
    unique_categories = set(categories)
    diversity_score   = len(unique_categories) / len(items)

    return round(diversity_score, 3)