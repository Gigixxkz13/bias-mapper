import re
from backend.analysis import calculate_sentiment, count_keywords, calculate_diversity, categorize_item
from backend.database import insert_response, insert_list_item


def extract_numbered_list_items(text):
    lines = text.splitlines()
    items = []

    for line in lines:
        line = line.strip()

        if re.match(r"^\d+\.", line):
            item = re.sub(r"^\d+\.\s*", "", line)
            items.append(item)

    return items


def process_and_store_response(run_id, identifier, prompt_text, raw_output_text):
    sentiment_score = calculate_sentiment(raw_output_text)

    keyword_counts = count_keywords(raw_output_text)
    risk_count = keyword_counts["riskCount"]
    benefit_count = keyword_counts["benefitCount"]
    emotion_count = keyword_counts["emotionCount"]

    list_items = extract_numbered_list_items(raw_output_text)

    if list_items:
        diversity_score = calculate_diversity(list_items)
    else:
        diversity_score = None

    response_id = insert_response(
        run_id=run_id,
        identifier=identifier,
        prompt_text=prompt_text,
        raw_output_text=raw_output_text,
        sentimentScore=sentiment_score,
        diversityScore=diversity_score,
        riskCount=risk_count,
        benefitCount=benefit_count,
        emotionCount=emotion_count
    )

    for index, item in enumerate(list_items, start=1):
        assigned_category = categorize_item(item)

        insert_list_item(
            response_id=response_id,
            numberOfItem=index,
            item_text=item,
            assigned_category=assigned_category
        )

    return {
        "response_id": response_id,
        "sentimentScore": sentiment_score,
        "diversityScore": diversity_score,
        "riskCount": risk_count,
        "benefitCount": benefit_count,
        "emotionCount": emotion_count,
        "listItemsFound": len(list_items)
    }