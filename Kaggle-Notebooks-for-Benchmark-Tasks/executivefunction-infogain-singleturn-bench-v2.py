#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We import the library as 'kbench' for brevity
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass

print("Ready to benchmark!")


# In[2]:


get_ipython().system('pip install datasets transformers torch pandas tabulate')


# In[3]:


from datasets import load_dataset
import urllib.request

import pandas as pd
import requests

# 1. Download the file once to your local machine/worker
url = "https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv"
local_path = "five_letter_words.csv"
urllib.request.urlretrieve(url, local_path)


# In[4]:


SYSTEM_PROMPT = """

You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- The game can start at any turn - may be with zero guess or few guesses made before.
- The guess must be a valid **5-letter English word**.
- For the guess you make, you will receive feedback indicating how close your guess was.
- Note that you can make only one guess per prompt requeest as to make next guess you would require feedback for current guess!

### Feedback Format:

Each letter in your guess will receive one of three symbols:

1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:

Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)


### Strategy & Reasoning Protocol

You must approach the game strategically by building rules, not as open-ended guessing. Every guess must be logically derived from prior feedback and should aim to maximize information gain while respecting all constraints.

Step-by-Step Reasoning Procedure

1.  Parse All Previous Feedback
    Read every prior guess and its feedback carefully. Treat each guess-feedback pair as a set of hard constraints on the secret word. Do not ignore earlier constraints unless logically required, such as in repeated-letter scenarios.

2.  Build a Global Constraint State
    From all previous turns, construct a unified constraint model:

    2.1 Confirmed Positions (✓)
    Identify letters that are fixed at exact indices. These positions are immutable and must remain unchanged in all future guesses.

    2.2 Misplaced Letters (-)
    Identify letters that must exist in the word but are not in the positions they were guessed. Track all invalid positions for each such letter.

    2.3 Eliminated Letters (x)
    Identify letters that are not present in the word. If a letter appears multiple times with mixed feedback, handle it carefully by considering frequency constraints.

3.  Maintain Positional Constraints
    For each of the five positions:

    3.1 Maintain a set of allowed letters
    3.2 Exclude letters marked as eliminated
    3.3 Exclude letters known to be invalid at that position from prior feedback

4.  Track Letter Inventory
    Maintain a consistent record of:

    4.1 Required letters (from ✓ and - feedback)
    4.2 Forbidden letters (from x feedback)
    4.3 Minimum or maximum occurrences of letters when repeated-letter patterns appear

5.  Vowel and Structural Analysis
    Ensure the candidate word is structurally valid:

    5.1 The word must contain at least one vowel (A, E, I, O, U, or Y)
    5.2 Prefer introducing new vowels if vowel information is uncertain
    5.3 Avoid repeating letters unless repetition is supported by prior feedback

6.  Candidate Generation
    Generate 2 to 3 candidate words that satisfy all constraints:

    6.1 Each candidate must be a valid 5-letter English word
    6.2 Each must satisfy all positional and letter constraints
    6.3 No candidate should include eliminated letters
    6.4 Required letters must appear only in valid positions

7. Candidate Evaluation (Critical Step)
   Among valid candidates, select based on:
   7.1 Finding the most clues by testing new letters
   7.2 Reducing the number of remaining possible words
   7.3 Balancing exploration (early guesses) and precision (later guesses)

8. Final Selection
   Choose exactly one final guess that:
   8.1 Fully satisfies all constraints
   8.2 Provides the best tradeoff between certainty and finding new clues

Important Behavioral Rules

1.  Never make a guess that violates any known constraint
2.  Never ignore prior feedback
3.  Never reuse eliminated letters
4.  Avoid random guessing; every guess must be logically justified
5.  If only one valid candidate remains, you must choose it

Example (Condensed)

Given:

1.  DEATH → D(x) E(x) A(x) T(x) H(-)
2.  SHIRK → S(✓) H(✓) I(x) R(x) K(x)

You must infer:

1.  S is in position 1 and H is in position 2
2.  H exists but is not in position 5
3.  Eliminated letters: D, E, A, T, I, R, K
4.  Generate valid candidates such as SHOWN, SHOWY, then select the best one

### Enforcement

1.  Before you start making a guess, you must first construct your rules and logical elimination inside the <think> tags.
2.  Your reasoning must reflect accumulated constraints across all turns
3.  Your final guess must be fully consistent with all derived rules
4.  Be concise in your reasoning. Ensure you arrive at a final <guess> within 1024 tokens. Accuracy and consistency with prior feedback are more important than the length of the explanation.
5.  Do not include spaces inside the guess tag. The word must be in ALL CAPS.

### CRITICAL OUTPUT RULE: 

You must output exactly ONE guess as per format below using `<guess>` tags at the very end of your response.

Example of expected final output structure:

<think>
[Your step-by-step logical deductions go here...]
</think>

<guess>BEACH</guess>

### Response Format:

Think through the problem and feedback step by step. Make sure to first frame the rules based on given previous feedback. Ensure that the step by step thought process is within <think> </think> tags. Then, return your guessed word in the following format: <guess> guessed-word </guess>.

"""


# In[5]:


import random
import pandas as pd
import json

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
# Set to an integer (e.g., 42) for fixed datasets across runs. 
# Set to None if you want a truly random dataset on every run.
RANDOM_SEED = 43 

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
# ==========================================

def generate_custom_feedback(guess, secret):
    """Generates Wordle feedback in the format: Letter(Symbol)"""
    guess = guess.upper()
    secret = secret.upper()

    feedback_symbols = ['x'] * len(secret)
    secret_letters = list(secret)
    guess_letters = list(guess)

    # First pass: Find exact matches (✓)
    for i in range(len(secret)):
        if guess_letters[i] == secret_letters[i]:
            feedback_symbols[i] = '✓'
            secret_letters[i] = None 
            guess_letters[i] = None

    # Second pass: Find partial matches (-)
    for i in range(len(secret)):
        if guess_letters[i] is not None and guess_letters[i] in secret_letters:
            feedback_symbols[i] = '-'
            secret_letters[secret_letters.index(guess_letters[i])] = None

    formatted_feedback = []
    for i in range(len(guess)):
        formatted_feedback.append(f"{guess[i].upper()}({feedback_symbols[i]})")

    return " ".join(formatted_feedback)

# --- Dataset Generation ---

# 1. Load your local word list
# (Assuming local_path is defined earlier in your script)
with open(local_path, "r", encoding="utf-8") as file:
    all_words = [line.strip().upper() for line in file if line.strip()]

new_secrets = random.sample(all_words, 200) 
mock_guesses = all_words 
full_word_list_url = "https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv"

# THIS IS THE KEY: Use a simple list to store your dictionaries (rows)
dataset_rows = []

for secret in new_secrets:
    past_guess_history = []

    # Because of the seed, it will always pick the exact same number of previous guesses
    available_guesses = [g for g in mock_guesses if g != secret]
    num_guesses = random.randint(1, 3) 

    # And it will always pick the exact same previous guess words
    previous_guesses = random.sample(available_guesses, num_guesses)

    for guess in previous_guesses:
        feedback = generate_custom_feedback(guess, secret)
        past_guess_history.append([guess, feedback])

    history_text = "\n".join([f"Guess: {g[0]} -> Feedback: {g[1]}" for g in past_guess_history])

    # Construct the combined prompt
    prompt = f"{SYSTEM_PROMPT}\n\nHere is some previous history:\n{history_text}"

    # Append each game as its own dictionary (matching Predibase structure)
    dataset_rows.append({
        "prompt": prompt, 
        "word_list": full_word_list_url,
        "past_guess_history": past_guess_history,        
        "secret": secret
    })

# --- FINAL ASSIGNMENT ---
dataset = dataset_rows

# --- Verification ---
print(f"Row 0 check - Secret should be a string: {dataset[0]['secret']}")

# Convert to DataFrame ONLY for the purpose of saving a file
df = pd.DataFrame(dataset)
df.to_json("wordle_eval_dataset2.jsonl", orient="records", lines=True)

print("Dataset successfully created. You can now run your testing loop.")


# In[6]:


df.info()


# In[7]:


import os
import re
import ast
import math
import pandas as pd
import torch
from enum import Enum
from typing import List
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ==========================================
# 1. HELPER FUNCTIONS & CLASSES
# ==========================================

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"

def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    valid_letters = set(secret_word)
    feedback = []
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)
        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)
        else:
            feedback.append(LetterFeedback.WRONG_LETTER)
    return feedback

@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"

    @staticmethod
    def from_secret(guess: str, secret: str) -> "GuessWithFeedback":
        return GuessWithFeedback(guess, get_feedback(guess, secret))


def guess_value(prompts, completions, **kwargs) -> list[float]:
    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        feedback = []
        secret_list = list(secret)
        for i, (g_char, s_char) in enumerate(zip(guess, secret)):
            if g_char == s_char:
                feedback.append(f"{g_char}(✓) ")
                secret_list[i] = None
            else:
                feedback.append(None)
        for i, g_char in enumerate(guess):
            if feedback[i] is None:
                if g_char in secret_list:
                    feedback[i] = f"{g_char}(-) "
                    secret_list[secret_list.index(g_char)] = None
                else:
                    feedback[i] = f"{g_char}(x) "
        if raw_feedback:
            return feedback
        return "".join(feedback).strip()

    def filter_candidates(all_candidate_words, past_guesses):
        filtered = []
        for word in all_candidate_words:
            valid = True
            for past_guess, past_feedback in past_guesses:
                candidate_feedback = validate_guess(word, past_guess)
                if candidate_feedback != past_feedback:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered

    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)
        if total_candidates == 0:
            return 0.0, 0.0
        current_entropy = math.log2(total_candidates)
        feedback_groups = {}
        for word in candidates:
            feedback = validate_guess(word, guess, raw_feedback=True)
            feedback_pattern = "".join('1' if "✓" in fb else ('0' if "-" in fb else 'x') for fb in feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)
        expected_entropy = 0
        max_info_gain = 0
        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            group_entropy = math.log2(group_size) if group_size > 0 else 0
            expected_entropy += p * group_entropy
            info_gain = current_entropy - group_entropy
            max_info_gain = max(max_info_gain, info_gain)
        expected_gain = current_entropy - expected_entropy
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0
        normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0
        return normalized_expected_gain, normalized_max_gain

    rewards = []
    word_lists = kwargs.get("word_list", [])
    past_guess_histories = kwargs.get("past_guess_history", [])

    for i in range(len(prompts)):
        try:
            comp = completions[i]
            completion_text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            completion_text = "<think>" + completion_text

            regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
            match = re.search(regex, completion_text, re.DOTALL)
            if not match or len(match.groups()) != 1:
                rewards.append(0.0)
                continue

            guess = match.groups()[0].strip()
            if len(guess) != 5:
                rewards.append(0.0)
                continue

            word_list_df = pd.read_csv(str(word_lists[i]))
            if guess not in word_list_df["Word"].values:
                rewards.append(0.0)
                continue

            past_guess_history = ast.literal_eval(past_guess_histories[i])
            normalized_expected_gain, _ = compute_normalized_information_gain(
                word_list_df["Word"].values,
                past_guess_history,
                guess
            )
            rewards.append(float(normalized_expected_gain))
        except Exception as e:
            rewards.append(0.0)

    return rewards

from typing import List

def get_messages(past_guesses: List[GuessWithFeedback]):
    return [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user", 
            "content": render_user_prompt(past_guesses)
        },
        {
            "role": "assistant", 
            # Simplified to just <think> so it doesn't hallucinate fake history
            "content": "Let me solve this step by step.\n<think>" 
        }
    ]


# In[8]:


def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    # THE FIX: Handle the empty state explicitly
    if not past_guesses:
        return "Make a new 5-letter word guess. This is the very first turn. There are no previous guesses. Do not assume any prior feedback."

    prompt = "Make a new 5-letter word guess.\n"
    prompt += "\nHere is the exact feedback from all previous guesses:\n"
    for i, past_guess in enumerate(past_guesses):
        prompt += f"Guess {i+1}: {past_guess}\n"
    return prompt


import re

def extract_guess(completion: str) -> str:
    # 1. Find ALL <guess> tags in the generated text
    matches = re.findall(r"<guess>\s*([\s\S]*?)\s*<\/guess>", completion, re.IGNORECASE)

    if matches:
        # Work backwards from the last match found (since the real guess is usually at the end)
        for match in reversed(matches):
            word = match.strip().upper()

            # Clean out any accidental punctuation or spaces the model added
            word = re.sub(r'[^A-Z]', '', word)

            # If it's exactly 5 letters, we found our target!
            if len(word) == 5:
                return word

        # If somehow none were 5 letters, return the very last one anyway to fail gracefully
        return matches[-1].strip().upper()

    # 2. Fallback: If no tags exist at all, grab the very last 5-letter word it mentioned
    fallback_matches = re.findall(r'\b[a-zA-Z]{5}\b', completion)
    if fallback_matches:
        return fallback_matches[-1].upper()

    return ""


# In[9]:


import pandas as pd
import json
import time
import kaggle_benchmarks as kbench
from pathlib import Path
import os
import glob

# Load datasets
WORKDIR = Path(".")
df = pd.read_json("wordle_eval_dataset2.jsonl", orient="records", lines=True)
word_list_df = pd.read_csv("five_letter_words.csv") # Update with your local path
allowed_words = word_list_df["Word"].str.upper().tolist()

# ==========================================
# 1. SINGLE ROW TASK
# ==========================================
@kbench.task(name="solve_single_wordle_v2")
def solve_single_wordle_v2(llm, prompt: str, past_guess_history: list, secret: str, **kwargs) -> dict:

    # Pace the API to prevent rate-limiting (NoneType errors)
    time.sleep(5) 

    try:
        completion = llm.prompt(prompt)
    except Exception as e:
        print(f"API Exception: {e}")
        return {"reward": 0.0, "info_gain": 0.0, "success_bonus": 0.0, "guess": "ERROR", "error": str(e)}

    guess = extract_guess(completion)

    # Handle Formatting Failures
    valid_format = len(guess) == 5 and guess in allowed_words
    kbench.assertions.assert_true(
        valid_format,
        expectation=f"The model must output a valid 5-letter word inside <guess> tags. Got: {guess}"
    )

    if not valid_format:
        return {"reward": 0.0, "info_gain": 0.0, "success_bonus": 0.0, "guess": guess, "error": "Invalid Format"}

    # Calculate Sub-Metrics
    history_for_reward = [(g[0], g[1]) for g in past_guess_history] 
    reward_output = guess_value(
        prompts=[prompt],
        completions=[completion], 
        word_list=["five_letter_words.csv"], 
        past_guess_history=[str(history_for_reward)]
    )

    info_gain = reward_output[0]
    success_bonus = 1.0 if guess == secret.upper() else 0.0
    total_reward = info_gain + success_bonus

    # Return a rich dictionary to be normalized later
    return {
        "reward": total_reward,
        "info_gain": info_gain,
        "success_bonus": success_bonus,
        "guess": guess,
        "error": "None"
    }

# ==========================================
# 2. ADVANCED BATCH EVALUATION TASK
# ==========================================
@kbench.task(name="evaluate_wordle_single_turn_v2") 
def evaluate_wordle_single_turn_v2(llm, df) -> float:
    with kbench.client.enable_cache():
        runs = solve_single_wordle_v2.evaluate(
            llm=[llm],
            evaluation_data=df,
            # max_attempts=5,
            # retry_delay=15,
            timeout=180,
            n_jobs=1, # Keep at 1 to avoid Gemini Rate Limits
            remove_run_files=True,
            stop_condition=lambda collected_runs: len(collected_runs) == df.shape[0],
        )

    results_df = runs.as_dataframe()
    if results_df.empty:
        raise RuntimeError("No benchmark runs were collected.") 

    expanded = pd.json_normalize(results_df["result"])
    expanded.index = results_df.index # Force indices to match
    combined = pd.concat([results_df.drop(columns=["result"]), expanded], axis=1) 

    # Save granular results to CSV so you can inspect individual failures
    combined.to_csv(WORKDIR / "wordle_executive_function_results.csv", index=False)

    # Calculate overall and sub-metrics
    overall_score = float(combined["reward"].mean())
    avg_info_gain = float(combined["info_gain"].mean())
    success_rate = float(combined["success_bonus"].mean())
    format_success_rate = float((combined["error"] == "None").mean())

    summary_payload = {
        "item_count": int(combined.shape[0]),
        "overall_score": overall_score,
        "average_info_gain": avg_info_gain,
        "win_rate": success_rate,
        "format_adherence_rate": format_success_rate
    }

    # Write summary to JSON
    (WORKDIR / "wordle_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2))

    # Kaggle Leaderboards usually track the single returned float
    return overall_score 


# In[10]:


evaluate_wordle_single_turn_v2.run(kbench.llm, df)


# In[11]:


get_ipython().run_line_magic('choose', 'evaluate_wordle_single_turn_v2')


# In[ ]:




