# utils/analyzer.py (Batch version)
import google.generativeai as genai
import re
import json
import os
from dotenv import load_dotenv
from utils.document_parser import extract_qa_pairs

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

def analyze_answers(document_text):
    qa_pairs = extract_qa_pairs(document_text)

    if not qa_pairs:
        return [{"error": "No question-answer pairs found in the document"}]

    model = genai.GenerativeModel("gemini-2.5-flash")

    # Build a batched prompt with all Q&A pairs
    batched_prompt = "You are an expert evaluator. Analyze the following Q&A pairs and return valid JSON array. Format for each:\n\n" + \
    """[
      {
        "question_num": "Q1",
        "question": "What is Python?",
        "answer": "Python is a programming language.",
        "is_correct": true,
        "correct_answer": "Python is a high-level programming language.",
        "explanation": "The answer is correct but can be more detailed.",
        "suggestion": "Mention that it's high-level and widely used for scripting, web dev, etc."
      }
    ]\n\n""" + \
    "Now analyze:\n"

    for pair in qa_pairs:
        batched_prompt += f"Q{pair['question_num']}: {pair['question']}\n"
        batched_prompt += f"A{pair['question_num']}: {pair['answer']}\n\n"

    try:
        response = model.generate_content(batched_prompt)
        response_text = response.text.strip()

        print(f"\n🔍 Gemini Raw Response (BATCH):\n{response_text[:1000]}\n")

        # Extract JSON array from response
        match = re.search(r'\[\s*{.*?}\s*\]', response_text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON array found in Gemini response.")

        parsed = json.loads(match.group(0))

        # ✅ Attach original answers back by matching question_num
        for item in parsed:
            q_num = item.get("question_num")
            original = next((q for q in qa_pairs if q["question_num"] == q_num), None)
            if original:
                item["answer"] = original["answer"]

            # Set defaults if keys are missing
            item.setdefault("question", "")
            item.setdefault("answer", "")
            item.setdefault("is_correct", False)
            item.setdefault("correct_answer", "")
            item.setdefault("explanation", "")
            item.setdefault("suggestion", "")

        print("\n✅ Final Parsed Analysis:")
        for item in parsed:
            print(json.dumps(item, indent=2))

        return parsed

    except Exception as e:
        print("❌ Error analyzing batch:", str(e))
        return [{
            "error": f"Batch analysis failed: {str(e)}"
        }]
