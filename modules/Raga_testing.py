


# import os
# import json
# import re
# import pandas as pd
# from tqdm import tqdm
# from dotenv import load_dotenv
# import google.generativeai as genai


# # ====================== CONFIGURATION ======================
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# EXCEL_PATH = "data/qa_with_predictions.xlsx"
# OUTPUT_XLSX = "data/qa_evaluated_raga_scores.xlsx"
# OUTPUT_JSON = "data/raga_testing_score.json"

# # Index range for partial evaluation
# START_INDEX = 572
# END_INDEX = START_INDEX + 20

# # ====================== INITIALIZATION ======================
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel("models/gemini-2.0-flash")


# # ====================== HELPERS ======================
# def build_prompt(question: str, answer: str, predicted_answer: str, context: str) -> str:
#     """Constructs the evaluation prompt aligned closely with RAGAS metric definitions."""
#     return f"""
# You are evaluating the quality of a predicted answer in a retrieval-augmented question-answering (RAG) system. Use the provided *context*, *question*, and *ground truth answer* to score the *predicted answer* on the following 4 criteria:

# Return ONLY a Python-style list of 4 floats between 0 and 1 (inclusive), rounded to 5 decimal places, and in this order:

# 1. Faithfulness: To what extent is the predicted answer strictly supported by the provided context? (1 = completely supported, 0 = contradicts or unsupported)
# 2. Answer Relevancy: How well does the predicted answer fully and directly address the question? (1 = fully answers the question, 0 = irrelevant)
# 3. Context Precision: What proportion of the retrieved context is actually relevant to generating the answer? (1 = all context is useful, 0 = none of the context is relevant)
# 4. Context Recall: How completely does the provided context support the ground truth answer? (1 = context fully contains the ground truth answer, 0 = context lacks key information)

# Important:
# - Use only the *provided context* to judge Faithfulness and Context Recall.
# - Do NOT include explanations, just return a Python-style list like this: [0.95521, 0.85312, 0.91230, 0.85428]

# Question: {question}
# Ground Truth Answer: {answer}
# Predicted Answer: {predicted_answer}
# Context: {context}
# """


# def extract_scores(response_text: str) -> list[float]:
#     """Extracts list of floats from Gemini's response."""
#     match = re.search(r"\[\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*\]", response_text)
#     return [float(match.group(i)) for i in range(1, 5)] if match else [None] * 4


# def evaluate_responses(df: pd.DataFrame) -> list[list[float]]:
#     """Generates scores for each row using Gemini."""
#     scores_list = []
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#         prompt = build_prompt(row["Question"], row["Answer"], row["Predicted_Answer"], row["Context"])
#         try:
#             response = model.generate_content(prompt)
#             scores = extract_scores(response.text)
#         except Exception as e:
#             print(f"❌ Error at index {START_INDEX + idx}: {e}")
#             scores = [None] * 4
#         print(f"✅ Row {START_INDEX + idx} scores: {scores}")
#         scores_list.append(scores)
#     return scores_list


# def append_to_excel(df_slice: pd.DataFrame):
#     """Appends evaluated results to Excel file."""
#     if os.path.exists(OUTPUT_XLSX):
#         existing_df = pd.read_excel(OUTPUT_XLSX)
#         combined_df = pd.concat([existing_df, df_slice], ignore_index=True)
#     else:
#         combined_df = df_slice
#     combined_df.to_excel(OUTPUT_XLSX, index=False)


# def update_json_scores(new_scores: list[list[float]]):
#     """Updates average scores JSON based on new and existing evaluations."""
#     filtered_scores = [s for s in new_scores if None not in s]
#     if not filtered_scores:
#         print("⚠️ No valid scores to update in JSON.")
#         return

#     new_count = len(filtered_scores)
#     new_sums = [sum(s[i] for s in filtered_scores) for i in range(4)]

#     old_sums, old_count = [0.0] * 4, 0
#     if os.path.exists(OUTPUT_JSON):
#         with open(OUTPUT_JSON, "r") as f:
#             existing_avg = json.load(f)
#         old_scores = [
#             existing_avg["faithfulness"],
#             existing_avg["answer_relevancy"],
#             existing_avg["context_precision"],
#             existing_avg["context_recall"],
#         ]
#         old_count = len(pd.read_excel(OUTPUT_XLSX)) - new_count
#         old_sums = [old_scores[i] * old_count for i in range(4)]

#     total_count = new_count + old_count
#     final_avg = {
#         "faithfulness": round((old_sums[0] + new_sums[0]) / total_count, 4),
#         "answer_relevancy": round((old_sums[1] + new_sums[1]) / total_count, 4),
#         "context_precision": round((old_sums[2] + new_sums[2]) / total_count, 4),
#         "context_recall": round((old_sums[3] + new_sums[3]) / total_count, 4),
#     }

#     with open(OUTPUT_JSON, "w") as f:
#         json.dump(final_avg, f, indent=2)

#     print("\n📊 Updated Evaluation Averages:")
#     print(json.dumps(final_avg, indent=2))


# # ====================== MAIN WORKFLOW ======================
# def main():
#     df = pd.read_excel(EXCEL_PATH)
#     df_slice = df.iloc[START_INDEX:END_INDEX].copy()

#     scores = evaluate_responses(df_slice)
#     df_slice["evaluation_scores"] = scores

#     append_to_excel(df_slice)
#     update_json_scores(scores)


# if __name__ == "__main__":
#     main()


import os
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# ====================== CONFIGURATION ======================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EXCEL_PATH = "data/qa_with_predictions.xlsx"
OUTPUT_XLSX = "data/qa_evaluated_raga_scores_2.xlsx"
OUTPUT_JSON = "data/raga_testing_score_2.json"

BATCH_SIZE = 15
SLEEP_SECONDS = 60

# ====================== INITIALIZATION ======================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")

# ====================== HELPERS ======================
def build_prompt(question: str, answer: str, predicted_answer: str, context: str) -> str:
    return f"""
You are evaluating the quality of a predicted answer in a retrieval-augmented question-answering (RAG) system. Use the provided *context*, *question*, and *ground truth answer* to score the *predicted answer* on the following 4 criteria:

Return ONLY a Python-style list of 4 floats between 0 and 1 (inclusive), rounded to 5 decimal places, and in this order:

1. Faithfulness: To what extent is the predicted answer strictly supported by the provided context? (1 = completely supported, 0 = contradicts or unsupported)
2. Answer Relevancy: How well does the predicted answer fully and directly address the question? (1 = fully answers the question, 0 = irrelevant)
3. Context Precision: What proportion of the retrieved context is actually relevant to generating the answer? (1 = all context is useful, 0 = none of the context is relevant)
4. Context Recall: How completely does the provided context support the ground truth answer? (1 = context fully contains the ground truth answer, 0 = context lacks key information)

Important:
- Use only the *provided context* to judge Faithfulness and Context Recall.
- Do NOT include explanations, just return a Python-style list like this: [0.95521, 0.85312, 0.91230, 0.85428]

Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {predicted_answer}
Context: {context}
"""

def extract_scores(response_text: str) -> list[float]:
    match = re.search(r"\[\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*,\s*(\d\.\d+)\s*\]", response_text)
    return [float(match.group(i)) for i in range(1, 5)] if match else [None] * 4

def evaluate_responses(df: pd.DataFrame, start_index: int) -> list[list[float]]:
    scores_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating rows {start_index}-{start_index+len(df)-1}"):
        prompt = build_prompt(row["Question"], row["Answer"], row["Predicted_Answer"], row["Context"])
        try:
            response = model.generate_content(prompt)
            scores = extract_scores(response.text)
        except Exception as e:
            print(f"❌ Error at index {start_index + idx}: {e}")
            scores = [None] * 4
        print(f"✅ Row {start_index + idx} scores: {scores}")
        scores_list.append(scores)
    return scores_list

def append_to_excel(df_slice: pd.DataFrame):
    if os.path.exists(OUTPUT_XLSX):
        existing_df = pd.read_excel(OUTPUT_XLSX)
        combined_df = pd.concat([existing_df, df_slice], ignore_index=True)
    else:
        combined_df = df_slice
    combined_df.to_excel(OUTPUT_XLSX, index=False)

def update_json_scores(new_scores: list[list[float]]):
    filtered_scores = [s for s in new_scores if None not in s]
    if not filtered_scores:
        print("⚠️ No valid scores to update in JSON.")
        return

    new_count = len(filtered_scores)
    new_sums = [sum(s[i] for s in filtered_scores) for i in range(4)]

    old_sums, old_count = [0.0] * 4, 0
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            existing_avg = json.load(f)
        old_scores = [
            existing_avg["faithfulness"],
            existing_avg["answer_relevancy"],
            existing_avg["context_precision"],
            existing_avg["context_recall"],
        ]
        old_count = len(pd.read_excel(OUTPUT_XLSX)) - new_count
        old_sums = [old_scores[i] * old_count for i in range(4)]

    total_count = new_count + old_count
    final_avg = {
        "faithfulness": round((old_sums[0] + new_sums[0]) / total_count, 4),
        "answer_relevancy": round((old_sums[1] + new_sums[1]) / total_count, 4),
        "context_precision": round((old_sums[2] + new_sums[2]) / total_count, 4),
        "context_recall": round((old_sums[3] + new_sums[3]) / total_count, 4),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_avg, f, indent=2)

    print("\n📊 Updated Evaluation Averages:")
    print(json.dumps(final_avg, indent=2))

# ====================== MAIN WORKFLOW ======================
def main():
    df = pd.read_excel(EXCEL_PATH)

    already_done = 0
    if os.path.exists(OUTPUT_XLSX):
        already_done = len(pd.read_excel(OUTPUT_XLSX))

    print(f"🔁 Resuming from index {already_done}")

    for i in range(already_done, len(df), BATCH_SIZE):
        df_slice = df.iloc[i:i + BATCH_SIZE].copy()
        if df_slice.empty:
            break

        scores = evaluate_responses(df_slice, i)
        df_slice["evaluation_scores"] = scores

        append_to_excel(df_slice)
        update_json_scores(scores)

        print(f"⏳ Sleeping {SLEEP_SECONDS}s before next batch...\n")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
