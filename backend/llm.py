import requests

def call_llm(prompt):

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()["choices"][0]["message"]["content"]