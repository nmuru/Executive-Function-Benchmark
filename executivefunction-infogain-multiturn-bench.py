#!/usr/bin/env python
# coding: utf-8

# #####  <span style="color:red"> Update (v2): Resolved a silent data-passing error in the multi-turn evaluation wrapper that caused info_gain to default to 0.0. The models are now rerun, and the leaderboard should now correctly reflect the Information Gain metric. Note that while single-turn benchmark already focusses only on info_gain as main metric, the multi-turn task benchmark includes both.</span>

# # Executive Function InfoGain: Multi-Turn Evaluation
# 
# Welcome to the **Multi-Turn** phase of the **Executive-Function-InfoGain-Benchmark**. 
# 
# ### Objective
# While the single-turn notebook isolated immediate reasoning, this task evaluates **Iterative Strategy** and **Recursive Executive Function**. The model is tasked with solving a Wordle puzzle from start to finish.
# 
# ### Why Multi-Turn?
# Multi-turn evaluation tests two critical skills that single-turn cannot:
# 1. **Accumulative Constraint Management:** The model must maintain a growing set of rules over several interactions without "hallucinating" or forgetting early feedback.
# 2. **Dynamic Strategy Shifting:** The model must decide when to "explore" (guess words with new letters to gain info) vs. when to "exploit" (attempt to solve the word).
# 
# ### Methodology
# In this task, the model plays through the game. Each response must include the `<think>` reasoning trace and the final `<guess>`. The benchmark tracks the model's efficiency in narrowing down the search space until the secret word is identified.
# 
# *This task uses the same high-entropy reward mechanism as the single-turn task, but aggregates performance across the entire game duration.*

# ### Let us now import the requisite libraries for the kaggle benchmark

# In[1]:


# We import the library as 'kbench' for brevity
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass

print("Ready to benchmark!")


# ### Install the required system dependencies for dataset manipulation, table formatting, and model execution.

# In[2]:


get_ipython().system('pip install datasets transformers torch pandas tabulate')


# ### Lexicon Preparation
# To evaluate the models fairly, we need a ground-truth list of valid 5-letter English words. This list allows the benchmark to verify if a model's guess is a "legal" Wordle move before calculating Information Gain or constraint violations. 
# 
# We pull a curated dataset of 5-letter words from a reliable repository and store it locally for high-speed access during the benchmark execution.

# ### Execution Scale Control
# The following cell defines the sample size for the evaluation. 
# 
# * **Test Mode:** A small batch of 10 samples used to verify that the API and scoring logic are functioning correctly without consuming significant credits.
# * **Production Mode:** A full batch of 40 samples to ensure statistical significance for the final leaderboard rankings.

# In[3]:


test_mode = False
n = 4 if test_mode else 40


# In[4]:


from datasets import load_dataset
import urllib.request

import pandas as pd
import requests

# 1. Download the file once to your local machine/worker
url = "https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv"
local_path = "five_letter_words.csv"
urllib.request.urlretrieve(url, local_path)


# ### System Instruction & Strategic Reasoning Protocol
# 
# Next cell gives an elaborate system prompt which teh game will use
# 
# One of the most significant challenges for LLMs in constraint-satisfaction tasks like Wordle is "attentional drift"—the tendency to lose track of early constraints (like Gray letters) as the game progresses. 
# 
# #### Why This Detailed Prompt?
# Through iterative testing, I discovered that even frontier models frequently ignore prior feedback unless forced to follow a structured deduction process. This system prompt was designed to:
# * **Mimic Agentic Workflows:** We have incorporated "Skills" (similar to `skills.md` used by agentic tools like Claude Code) directly into the system instructions.
# * **Externalize Working Memory:** By requiring the model to build a "Global Constraint State" inside `<think>` tags, we force it to physically write out the rules before committing to a guess.
# * **Test Faithful Instruction Following:** This serves as a secondary benchmark—can the model adhere to a complex, multi-step logical procedure without skipping steps?
# 
# The prompt enforces a **8-step Reasoning Procedure**, covering feedback parsing, positional analysis, and candidate evaluation. The final output is strictly wrapped in `<guess>` tags to ensure programmatic parsability.

# In[5]:


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


# Next cell creates the multi-turn dataset with an emphasis on **Adversarial Difficulty**.
# 
# ### Multi-Turn Dataset Synthesis & Difficulty Engineering
# 
# The transition from single-turn to multi-turn evaluation revealed a critical benchmarking challenge: **Saturation.** To effectively rank state-of-the-art models, the dataset had to evolve through three distinct stages of difficulty.
# 
# #### The Evolution of the Word Pool
# * **The NYT Limitation:** During initial experimentation, I tested the standard NYT-approved 2,500-word list. The results showed models achieving win rates as high as **95%** with average scores reaching **1.7**. This suggested that the restricted word pool was too "solvable" for modern LLMs, leaving no room for strategic differentiation.
# * **The NLTK/Dictionary Attempt:** I briefly shifted to a full NLTK-based dictionary. While this solved the "easy" problem, it introduced a "fairness" problem—many 5-letter words were too obscure or archaic, testing vocabulary depth rather than executive reasoning.
# * **The "Scribble" Dataset (The Final Choice):** We settled on a global "Scribble" style dataset (aligned with Predibase methodologies). This provides a vast, high-entropy search space that prevents brute-forcing while remaining within the realm of recognizable language.
# 
# #### Adversarial Strategy: The "Rabbit Hole"
# To truly test executive function, I specifically curated scenarios that lead models into **"logical traps"** (also known as rhyme families or high-ambiguity clusters).
# * **Strategic Exploration:** By choosing secrets like *LIGHT* (which shares a pattern with *MIGHT, SIGHT, FIGHT, NIGHT*), the model is forced to abandon blind guessing.
# * **Information Theory over Luck:** Success in these "rabbit holes" requires the model to utilize **Information Gain**—guessing a word like *FORMS* to eliminate multiple candidates at once rather than guessing individual rhyme members.
# 
# The dataset simulates game states with **1 to 3 prior guesses**, requiring the model to maintain a growing set of logical constraints and drive the game to a successful conclusion.

# In[6]:


import random
import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================
# 1. CORE UTILITIES
# ==========================================
GLOBAL_WORD_LIST_URL = "https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv"


global_df = pd.read_csv(GLOBAL_WORD_LIST_URL)
all_allowed_guesses = sorted(global_df['Word'].str.upper().tolist())

def generate_custom_feedback(guess, secret):
    feedback_symbols = ['x'] * 5
    secret_list = list(secret)
    guess_list = list(guess)
    for i in range(5):
        if guess_list[i] == secret_list[i]:
            feedback_symbols[i] = '✓'; secret_list[i] = None; guess_list[i] = None
    for i in range(5):
        if guess_list[i] is not None and guess_list[i] in secret_list:
            feedback_symbols[i] = '-'; secret_list[secret_list.index(guess_list[i])] = None
    return " ".join([f"{guess[i]}({feedback_symbols[i]})" for i in range(5)])

# ==========================================
# 2. DETERMINISTIC GENERATORS
# ==========================================
def create_standard_dataset(n_samples, seed=43):
    # HARD RESET SEED INSIDE FUNCTION
    random.seed(seed) 

    local_pool = sorted(all_allowed_guesses.copy())
    selected_secrets = random.sample(local_pool, min(n_samples, len(local_pool)))

    count_1, count_2 = int(n_samples * 0.8), int(n_samples * 0.1)
    count_3 = n_samples - (count_1 + count_2)
    distribution_pool = [1] * count_1 + [2] * count_2 + [3] * count_3
    random.shuffle(distribution_pool)

    dataset_rows = []
    for i, secret in enumerate(selected_secrets):
        past_guess_history = []
        current_valid_pool = sorted([w for w in all_allowed_guesses if w != secret])
        num_guesses = distribution_pool[i]
        for _ in range(num_guesses):
            if not current_valid_pool: break
            guess = random.choice(current_valid_pool)
            feedback = generate_custom_feedback(guess, secret)
            past_guess_history.append([guess, feedback])
            current_valid_pool = sorted([w for w in current_valid_pool if generate_custom_feedback(guess, w) == feedback and w != secret])

        # --- FIX ADDED HERE ---
        history_text = "No previous history."
        if past_guess_history:
            history_text = "Previous History:\n" + "\n".join(
                [f"Guess {idx+1}: {g[0]} -> Feedback: {g[1]}" for idx, g in enumerate(past_guess_history)]
            )
        # ----------------------

        dataset_rows.append({
            "prompt": f"{SYSTEM_PROMPT}\n\n{history_text}\n\nPlease provide your next guess.", 
            "word_list": GLOBAL_WORD_LIST_URL, "past_guess_history": past_guess_history, "secret": secret, "puzzle_type": "standard"
        })
    return pd.DataFrame(dataset_rows)

def create_difficult_dataset(n_puzzles, seed=43):
    # HARD RESET SEED INSIDE FUNCTION 
    # This ensures Block 2 doesn't care what happened in Block 1
    random.seed(seed)

    patterns = defaultdict(list)
    for word in sorted(all_allowed_guesses):
        for i in range(5):
            pattern = word[:i] + '*' + word[i+1:]; patterns[pattern].append(word)

    dense_neighborhoods = {k: sorted(v) for k, v in patterns.items() if len(v) >= 5}
    sorted_neighborhoods = sorted(dense_neighborhoods.items(), key=lambda x: (len(x[1]), x[0]), reverse=True)

    dataset_rows = []
    for pattern, neighbors in sorted_neighborhoods[:n_puzzles]:
        secret = random.choice(neighbors)
        similar_guess = random.choice(sorted([w for w in neighbors if w != secret]))
        exploration_start = random.choice(sorted([w for w in all_allowed_guesses if w not in neighbors]))

        past_guess_history = [[exploration_start, generate_custom_feedback(exploration_start, secret)], 
                              [similar_guess, generate_custom_feedback(similar_guess, secret)]]

        dataset_rows.append({
            "prompt": f"{SYSTEM_PROMPT}\n\nPrevious History:\n" + "\n".join([f"Guess {i+1}: {g[0]} -> Feedback: {g[1]}" for i, g in enumerate(past_guess_history)]),
            "word_list": GLOBAL_WORD_LIST_URL, "past_guess_history": past_guess_history, "secret": secret, "puzzle_type": "high_difficulty"
        })
    return pd.DataFrame(dataset_rows)

# ==========================================
# 3. EXECUTION
# ==========================================

df_standard = create_standard_dataset(n_samples=n, seed=42)
df_difficult = create_difficult_dataset(n_puzzles=int(0.3 * n), seed=42)

original_columns = ["prompt", "word_list", "past_guess_history", "secret"]
df_merged = pd.concat([df_standard[original_columns], df_difficult[original_columns]], ignore_index=True)

# Final deterministic shuffle
df_merged = df_merged.sample(frac=1, random_state=43).reset_index(drop=True)


# In[7]:


df_merged


# In[8]:


df_merged['prompt'][0]


# Next cell is used to ascertain the total number of API calls required for the evaluation.
# 
# ### **API Workload & Resource Estimation**
# 
# A critical distinction between the single-turn and multi-turn benchmarks is the **computational intensity**. 
# 
# * **Single-Turn:** Operates on a predictable 1:1 ratio (e.g., 75 rows = 75 API calls).
# * **Multi-Turn:** Operates on a variable 1:N ratio. Because each game represents a full logical sequence, a single dataset row can trigger multiple sequential LLM calls as the model iterates toward the solution.
# 
# #### **Workload Scaling**
# In this evaluation, we simulate games that have already progressed through 1 to 3 turns. This means each game has a potential "tail" of up to 5 remaining attempts. For a production run of 40 games, the workload can easily scale to **200+ individual API calls**. 
# 
# This cell calculates the "Maximum LLM Calls" to ensure the following:
# 1. **Credit Allocation:** Verification that the current run fits within the allocated API budget.
# 2. **Time Management:** Providing an estimate of total execution time, given the 5-second pacing delay between calls.
# 3. **Pacing Strategy:** Ensuring we do not exceed rate limits when the model is deep in a "Rabbit Hole" and making rapid sequential guesses.
# 
# 

# In[9]:


# Create temporary series for calculation
temp_prefilled = df_merged['past_guess_history'].apply(len)
temp_remaining = 6 - temp_prefilled

# Calculate Grand Total
grand_total_calls = temp_remaining.sum()

# Generate the Summary table without modifying df_merged
print("--- Workload by Group (One-Off Investigation) ---")
investigation_summary = pd.DataFrame({
    'prefilled_count': temp_prefilled,
    'calls_remaining': temp_remaining
}).groupby('prefilled_count').agg(
    row_count=('calls_remaining', 'count'),
    calls_per_row=('calls_remaining', 'first'),
    total_calls_for_group=('calls_remaining', 'sum')
)

print(investigation_summary)
print("-" * 35)
print(f"Grand Total Max LLM Calls: {grand_total_calls}")


# In[10]:


output_filename = "wordle_eval_dataset1.jsonl"
df_merged.to_json(output_filename, orient="records", lines=True)

print(f"Success! {len(df_merged)} rows saved with original column structure.")
print(f"Columns in final file: {df_merged.columns.tolist()}")


# Next cell contains the helper functions and the core reward mechanism.
# 
# ### The Heart of the Benchmark: Information Gain & Entropy
# 
# The most critical aspect of this evaluation is the **Information Theory** approach to scoring. Standard Wordle benchmarks often only look at whether a model found the secret word. However, in a strategic context, the *quality* of a guess is determined by how much it reduces uncertainty.
# 
# #### How Information Gain Captures Skill:
# * **Entropy ($\text{H}$):** We treat the set of all possible remaining words as a probability distribution. If there are 1024 possible words, the entropy is $10 \text{ bits}$ ($log_2(1024)$).
# * **Expected Information Gain:** A "smart" model doesn't just guess a random valid word; it selects a word that, regardless of the feedback received, will eliminate the maximum number of incorrect candidates. 
# * **Capturing Nuance:** This metric rewards models that use "exploratory" words (testing 5 new letters) early in the game, even if those words couldn't possibly be the secret answer. It distinguishes a model that is "thinking" strategically from one that is just guessing blindly.
# * **Normalization:** We normalize the gain against the starting entropy. This allows us to fairly compare models across different game states—a model that reduces a 500-word space to 10 words is scored as more capable than one that reduces a 4-word space to 2.
# 
# The following cell implements these mathematical principles into a programmatic `guess_value` function used by the `kbench` evaluator.

# In[11]:


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



import pandas as pd
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


# In[12]:


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

def extract_guess(completion_text: str) -> str:
    """Extracts the 5-letter guess, handling multiple tags, but STRICTLY enforces formatting."""
    completion = str(completion_text)

    # 1. Find ALL <guess> tags in the generated text ([\s\S] handles line breaks inside tags)
    matches = re.findall(r"<guess>\s*([\s\S]*?)\s*</guess>", completion, re.IGNORECASE)

    if matches:
        # Work backwards from the last match found (the model's final decision is usually at the end)
        for match in reversed(matches):
            # Clean out any accidental punctuation, spaces, or symbols the model hallucinated
            word = re.sub(r'[^A-Za-z]', '', match)
            word = word.upper()

            # If it's exactly 5 letters, we found our perfect target!
            if len(word) == 5:
                return word

        # If tags existed but none contained a valid 5-letter word, return the last one cleaned
        # so the main loop can correctly log "Invalid Format: [Word]" and apply the -0.3 penalty.
        return re.sub(r'[^A-Za-z]', '', matches[-1]).upper()

    # 2. STRICT BENCHMARK ENFORCEMENT
    # No fallback! If they forgot the tags entirely, they fail the format check.
    return "MISSING_TAG"


# In[13]:


import pandas as pd
import ast
import re
import math

def guess_value(prompts, completions, **kwargs) -> list[float]:

    # 1. FIXED: Sync validate_guess to enforce uppercase like your main loop
    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        secret = secret.upper()
        guess = guess.upper()
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
                if validate_guess(word, past_guess) != past_feedback:
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

    cached_word_lists = {}

    for i in range(len(prompts)):
        try:
            comp = completions[i]
            completion_text = comp[0]["content"] if isinstance(comp, list) else str(comp)

            # 2. FIXED: Use the exact same extractor as your main loop to prevent mismatches
            # Note: extract_guess is defined later in your notebook, but since Python 
            # evaluates functions at runtime, it is perfectly safe to call it here!
            guess = extract_guess(completion_text)

            if guess == "MISSING_TAG" or len(guess) != 5:
                rewards.append(0.0)
                continue

            # 3. FIXED: Force dictionary cache to be uppercase strings
            file_path = str(word_lists[i])
            if file_path not in cached_word_lists:
                df = pd.read_csv(file_path)
                words_list = df["Word"].astype(str).str.upper().tolist() 
                words_set = set(words_list)
                cached_word_lists[file_path] = (words_list, words_set)

            all_words_list, allowed_words_set = cached_word_lists[file_path]

            if guess not in allowed_words_set:
                rewards.append(0.0)
                continue

            history_raw = past_guess_histories[i]
            past_guess_history = ast.literal_eval(history_raw) if isinstance(history_raw, str) else history_raw

            normalized_expected_gain, _ = compute_normalized_information_gain(
                all_words_list,
                past_guess_history,
                guess
            )
            rewards.append(float(normalized_expected_gain))

        except Exception as e:
            rewards.append(0.0)

    return rewards


# Next cell implements the core evaluation logic and batch processing.
# 
# ### Evaluation Engine: Single-Turn Batch Processing
# 
# This section defines how the benchmark actually executes. Unlike a standard script, we use a **hierarchical task structure** within the `kbench` framework to ensure robustness and detailed logging.
# 
# #### Core Logic & Scoring Strategy:
# * **The "Solve" Unit:** We define a single-row task (`solve_single_wordle`) that handles the interaction with the LLM. It includes a mandatory **pacing delay (time.sleep)** to respect API rate limits, which is essential when testing high-tier models.
# * **Validation Assertions:** We use `kbench.assertions` to enforce that models produce a valid 5-letter word within the correct tags. Failure to do so results in an immediate zero for that row, punishing poor instruction following.
# 
# #### The Reward Mechanism (Information Theory Deep-Dive):
# The reward is not a simple "correct/incorrect" binary. To truly measure **Executive Function**, we calculate the **Normalized Information Gain** for every guess made.
# 
# 1.  **Expected Information Gain:** For every guess, the code calculates the **Entropy** of the remaining possible word set before and after the guess.
#     * If a model guesses a word that could result in many different feedback patterns (splitting the word pool into many small groups), it has high **Expected Information Gain**. 
#     * This rewards strategic "exploratory" play, even if the guess itself isn't the final secret word.
# 2.  **The Success Bonus:** While Information Gain measures strategy, the ultimate goal is the solution. A **+1.0 bonus** is added to the reward if the model successfully identifies the secret word.
# 3.  **Strict Penalties (The Zero-Score Rule):** A model receives a **0.0 reward** for a turn if:
#     * It fails to follow the `<guess>` tag format.
#     * It outputs a word that is not in the allowed 5-letter lexicon.
#     * The output word violates known hard constraints (e.g., using a letter already marked as 'Gray').
# 
# #### Granular Reporting:
# The batch evaluator (`score_wordle_executive_function`) generates a detailed CSV and JSON summary. This allows us to inspect not just the "win rate," but also the **Format Adherence Rate**, distinguishing between models that are "smart but messy" and those that are "precise but unstrategic."

# In[14]:


import pandas as pd
import json
import time
import re
import ast
import math
import kaggle_benchmarks as kbench
from pathlib import Path

# ==========================================
# SETUP & DATA LOADING
# ==========================================
WORKDIR = Path(".")
df = pd.read_json("wordle_eval_dataset1.jsonl", orient="records", lines=True)
word_list_df = pd.read_csv("five_letter_words.csv") 
allowed_words = word_list_df["Word"].str.upper().tolist()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
    feedback = []
    secret_list = list(secret.upper())
    guess = guess.upper()
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
    if raw_feedback: return feedback
    return "".join(feedback).strip()

def extract_think_block(completion_text: str) -> str:
    match = re.search(r"<guess>", str(completion_text), re.IGNORECASE)
    if match: return str(completion_text)[:match.start()].strip()
    return str(completion_text).strip()



def count_constraint_violations(current_guess: str, past_history: list) -> int:
    violations = 0
    for past_word, past_fb in past_history:
        if validate_guess(secret=current_guess, guess=past_word) != past_fb:
            violations += 1
    return violations


# ==========================================
# 1. SINGLE ROW TASK
# ==========================================
@kbench.task(name="solve_single_wordle_multiturn_v4")
def solve_single_wordle_multiturn_v4(llm, prompt: str, past_guess_history: list, secret: str, **kwargs) -> dict:
    current_history = [(g[0], g[1]) for g in past_guess_history]
    turns_left = 6 - len(current_history)
    current_prompt = prompt 
    game_results = []

    for _ in range(turns_left):
        curr_turn = len(current_history) + 1
        guesses_left = 6 - len(current_history) 

        if guesses_left == 1:
            state_tag = f"\n[SYSTEM STATE: Turn {curr_turn} of 6. WARNING: This is your final guess!]\n"
        else:
            state_tag = f"\n[SYSTEM STATE: Turn {curr_turn} of 6. You have {guesses_left} guesses available (including this one).]\n"

        # TRAP 1: API / NETWORK FAILURES
        try:
            completion = llm.prompt(current_prompt + state_tag)
            if not completion: raise Exception("Empty response from API")
        except Exception as e:
            game_results.append({
                "secret": secret, "turn": curr_turn, "guess": "ERROR", 
                "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0, "violations": 0,
                "error": f"API Exception: {str(e)[:100]}", "think_word_count": 0
            })
            break

        think_text = extract_think_block(completion)
        guess = extract_guess(completion)

        # TRAP 2: FORMATTING FAILURES (No -0.3 Penalty here anymore!)
        if guess == "MISSING_TAG" or len(guess) != 5 or guess not in allowed_words:
            error_reason = "Missing <guess> tag" if guess == "MISSING_TAG" else f"Invalid Format/Word: {guess}"
            game_results.append({
                "secret": secret, "turn": curr_turn, "guess": guess[:10], 
                "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0, "violations": 0, # Penalty is 0.0!
                "error": error_reason, "think_word_count": len(think_text.split())
            })
            break 

        # NORMAL PROCESSING: LOGIC & PENALTIES
        v = count_constraint_violations(guess, current_history)

        # STRICT LOGIC PENALTY APPLIED HERE
        if v > 0:
            info_gain = 0.0
            turn_penalty = -0.3 # Flat penalty applied for breaking logical rules
        else:
            info_gain = guess_value([current_prompt], 
                                    [completion], 
                                    past_guess_history=[current_history], 
                                    word_list=[kwargs.get("word_list")]
                                )[0]
            # info_gain = guess_value([current_prompt], [completion], past_guess_history=[current_history])[0]


            turn_penalty = 0.0 # No penalty, pure info gain

        entry = {
            "secret": secret, "turn": curr_turn, "guess": guess, 
            "info_gain": info_gain, "penalty": turn_penalty, "success_bonus": 0.0,
            "violations": v, "think_word_count": len(think_text.split()), "error": "None"
        }

        if guess == secret.upper():
            entry["success_bonus"] = 1.0 
            game_results.append(entry); break

        game_results.append(entry)
        current_history.append((guess, validate_guess(secret, guess)))
        current_prompt += f"\nAssistant: <guess>{guess}</guess>\nUser: Feedback: {validate_guess(secret, guess)}\n"

    return {"game_turns": game_results}

# ==========================================
# 2. BATCH EVALUATION
# ==========================================
@kbench.task(name="evaluate_wordle_multi_turn")
def evaluate_wordle_multi_turn(llm, df) -> float:
    with kbench.client.enable_cache():
        runs = solve_single_wordle_multiturn_v4.evaluate(llm=[llm], evaluation_data=df, n_jobs=1, remove_run_files=True)

    results_df = runs.as_dataframe()
    if results_df.empty: return 0.0

    combined = results_df.copy()
    combined['res'] = combined['result'].apply(lambda x: x.get('game_turns', []))
    combined = combined.explode("res").reset_index(drop=True)

    expanded = pd.json_normalize(combined["res"])
    if expanded.empty or 'error' not in expanded.columns:
        print("Error: No valid turn data collected. Check LLM connectivity."); return 0.0

    combined = pd.concat([combined.drop(columns=["result", "res"]), expanded], axis=1)

    # # --- ERROR THRESHOLD LOGIC ---
    # total_games = len(df) 
    # error_games = combined[combined['error'].str.contains('API Exception', na=False)]['id'].nunique()
    # error_rate = error_games / total_games if total_games > 0 else 1.0

    # valid = combined[~combined['error'].str.contains('API Exception', na=False)].copy()


    # --- ERROR THRESHOLD LOGIC ---
    total_games = len(df) 

    # 1. Get the exact IDs of the games that had API errors
    error_games_ids = combined[combined['error'].str.contains('API Exception', na=False)]['id'].unique()

    # 2. Calculate the error rate
    error_rate = len(error_games_ids) / total_games if total_games > 0 else 1.0

    # 3. THE FIX: Drop the ENTIRE game (all turns) if its ID is in the error list
    valid = combined[~combined['id'].isin(error_games_ids)].copy()



    if valid.empty: 
        combined.to_csv(WORKDIR / "wordle_full_logs.csv", index=False)
        summary = {
            "overall_benchmark_score": 0.0, "raw_score_before_penalty": 0.0,
            "win_rate": 0.0, "avg_violations_per_game": 0.0,
            "api_error_rate": float(error_rate), "penalty_applied": True
        }
        (WORKDIR / "wordle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0.0

    valid = valid.loc[:, ~valid.columns.duplicated()].copy()

    # --- STRICT AGGREGATION MATH ---
    game_grouped = valid.groupby('id').agg(
        total_info_gain=('info_gain', 'sum'),
        total_penalty=('penalty', 'sum'), # Flat sum of penalties
        llm_turns_played=('turn', 'count'), 
        violations=('violations', 'sum'),
        think_word_count=('think_word_count', 'mean'),
        success_bonus=('success_bonus', 'max'), 
        guess=('guess', 'last'),
        secret=('secret', 'first')
    )

    # TRUE MATH: Avg(Info Gain) + Sum(Penalty) + Max(Success Bonus)
    game_grouped['is_win'] = game_grouped['success_bonus']
    game_grouped['avg_info_gain'] = game_grouped['total_info_gain'] / game_grouped['llm_turns_played'].replace(0, 1)
    game_grouped['game_score'] = game_grouped['avg_info_gain'] + game_grouped['total_penalty'] + game_grouped['success_bonus']

    # --- CSV QUALITY OF LIFE INJECTION ---
    # Inject 'game_score' column into combined DF, but only place the value on the very last row for each ID
    combined['game_score'] = ""
    last_indices = combined.groupby('id').tail(1).index
    combined.loc[last_indices, 'game_score'] = combined.loc[last_indices, 'id'].map(game_grouped['game_score'].round(4))

    # Save the polished CSV
    combined.to_csv(WORKDIR / "wordle_full_logs.csv", index=False)

    # --- FINAL SCORE LOGIC ---
    raw_score = float(game_grouped['game_score'].mean())
    MAX_ALLOWED_ERROR_RATE = 0.20  

    if error_rate > MAX_ALLOWED_ERROR_RATE:
        final_score = 0.0
        penalty_flag = True
    else:
        final_score = raw_score
        penalty_flag = False

    summary = {
        "overall_benchmark_score": final_score,
        "raw_score_before_penalty": raw_score,
        "win_rate": float(game_grouped['is_win'].mean()),
        "avg_violations_per_game": float(game_grouped['violations'].mean()),
        "api_error_rate": float(error_rate),
        "penalty_applied": penalty_flag
    }

    (WORKDIR / "wordle_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return summary["overall_benchmark_score"]


# In[15]:


evaluate_wordle_multi_turn.run(kbench.llm,df)


# In[16]:


kbench.llm


# In[17]:


get_ipython().run_line_magic('choose', 'evaluate_wordle_multi_turn')

