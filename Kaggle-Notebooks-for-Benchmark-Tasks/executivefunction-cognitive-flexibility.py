#!/usr/bin/env python
# coding: utf-8

# # Executive Function InfoGain: Cognitive Flexibility
# 
# #### <span style="color: green;">Note: This notebook on measuring cognitive flexibility is the same as <span style="color: blue;">the notebook on the multi-turn Wordle benchmark</span>, except for some changes in the system prompt and scoring methodology: In the system prompt, we simply add these two instructions which entirely change the performance of the AI models on the Wordle game:</span>
# 
# #### (Note: Additionally, the percentage of 'difficult dataset' is increased to 80% and for exploration moves, information gain score is not given)
# 
# <div style="color: blue; line-height: 1.6;">
# "Your cognitive flexibility is also evaluated in this game. To succeed, you must know when to narrow down options and when to pivot your strategy. You have two valid types of moves:
# 
# <ul>
#     <li><b>Constraint Move (N):</b> Your guess perfectly satisfies all known hints from previous turns.<br>
#     <i>Output:</i> <code>&lt;guess-word,N&gt;</code></li>
#     <br>
#     <li><b>Exploration Move (Y):</b> You deliberately ignore some previous hints in order to test an entirely new batch of letters. This is a highly encouraged strategic sacrifice when you are trapped by too many similar valid words (e.g., you know the word ends in _IGHT, but R, M, L, N, and S are all untested).<br>
#     <i>Output:</i> <code>&lt;guess-word,Y&gt;</code></li>
# </ul>
# 
# <b>How is the score calculated?</b><br>
# The game evaluates your executive functions capability including cognitive flexibility and efficiency using three components: <b>Information Gain</b>, <b>Penalty</b>, and <b>Success Bonus</b>.
# 
# <ul>
#     <li><b>Information Gain:</b> Measures how effectively your guess reduces the pool of possible secret words. High uncertainty reduction equals high Information Gain.</li>
#     <li><b>The Exploration Mechanic (No Penalty for 'Y'):</b> If you make an Exploration Move (<code>&lt;guess-word,Y&gt;</code>), you will receive a <b>0.0 penalty</b>, AND you will earn the full Information Gain for the new clues your guess uncovers. This proves your cognitive flexibility by strategically gathering new clues to secure a win on a later turn.</li>
#     <li><b>The Hallucination Penalty (-0.3 for false 'N'):</b> You will receive a severe penalty of <b>-0.3</b> ONLY IF your guess violates a past constraint but you incorrectly claim it is a perfect Constraint Move by outputting <code>&lt;guess-word,N&gt;</code>.</li>
#     <li><b>Standard Play (No Penalty for valid 'N'):</b> If you perfectly follow all constraints and output <code>&lt;guess-word,N&gt;</code>, your penalty is <b>0.0</b> and you earn your full calculated Information Gain.</li>
# </ul>
# </div>

# ### Let us now import the requisite libraries for the kaggle benchmark

# In[ ]:


# We import the library as 'kbench' for brevity
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass

print("Ready to benchmark!")


# ### Install the required system dependencies for dataset manipulation, table formatting, and model execution.

# In[ ]:


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

# In[ ]:


test_mode = False
n =5 if test_mode else 40


# In[ ]:


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

# In[ ]:


# SYSTEM_PROMPT = """

# You are playing Wordle, a word-guessing game.

# ### Game Rules:
# - You have **6 tries** to guess a secret **5-letter** word.
# - Each guess must be a valid **5-letter English word**.
# - After each guess, you will receive feedback indicating how close your guess was.

# ### Feedback Format:
# Each letter in your guess will receive one of three symbols:
# 1. ✓ : The letter is in the word and in the CORRECT position.
# 2. - : The letter is in the word but in the WRONG position.
# 3. x : The letter is NOT in the word.

# ### Example:
# Secret Word: BRISK

# Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
# Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
# Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)




# ### CRITICAL OUTPUT RULE: 

# You must output exactly ONE guess as per format below using `<guess>` tags at the very end of your response.

# Example of expected final output structure:

# <think>
# [Your step-by-step logical deductions go here...]
# </think>

# <guess>BEACH</guess>

# ### Response Format:

# Think through the problem and feedback step by step. Make sure to first frame the rules based on given previous feedback. Ensure that the step by step thought process is within <think> </think> tags. Then, return your guessed word in the following format: <guess> guessed-word </guess>.

# """


# In[ ]:


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

### Move Types & Output Format

Your cognitive flexibility is also evaluated in this game. To succeed, you must know when to narrow down options and when to pivot your strategy. You have two valid types of moves:

1. **Constraint Move (N):** Your guess perfectly satisfies all known hints from previous turns. 
   - Output: `<guess-word,N>`
2. **Exploration Move (Y):** You deliberately ignore some previous hints in order to test an entirely new batch of letters. This is a highly encouraged strategic sacrifice when you are trapped by too many similar valid words (e.g., you know the word ends in `_IGHT`, but `R`, `M`, `L`, `N`, and `S` are all untested). 
   - Output: `<guess-word,Y>`

### How is the score calculated?

The game evaluates your executive functions capability including cognitive flexibility and efficiency using three components: Information Gain, Penalty, and Success Bonus.


1. **Information Gain:** Measures how effectively your guess reduces the pool of possible secret words. High uncertainty reduction equals high Information Gain.
2. **The Exploration Mechanic (No Penalty for 'Y'):** If you make an Exploration Move (`<guess-word,Y>`), you will receive **0.0 penalty**, AND you will earn the full Information Gain for the new clues your guess uncovers. This proves your cognitive flexibility by strategically gathering new clues to secure a win on a later turn.
3. **The Hallucination Penalty (-0.3 for false 'N'):** You will receive a severe penalty of -0.3 ONLY IF your guess violates a past constraint but you incorrectly claim it is a perfect Constraint Move by outputting `<guess-word,N>`. 
4. **Standard Play (No Penalty for valid 'N'):** If you perfectly follow all constraints and output `<guess-word,N>`, your penalty is 0.0 and you earn your full calculated Information Gain.

### Enforcement

1.  Before you start making a guess, you must first construct your rules and logical elimination inside the `<think>` tags.
2.  If you realize you are trapped by too many similar words, explicitly state your intent to use an Exploration Move in your `<think>` block.
3.  Be concise in your reasoning. Ensure you arrive at a final `<guess>` within 1024 tokens. 
4.  Do not include spaces inside the guess tag. The word must be in ALL CAPS.

### CRITICAL OUTPUT RULE: 

You must output exactly ONE guess as per the format below using `<guess>` tags at the very end of your response.

You MUST specify your move type by appending `,Y` or `,N` to your guessed word.

Example of expected final output structure:

<think>
[Your step-by-step logical deductions go here. Evaluate if an Exploration Move is necessary...]
</think>

### If you perfectly followed all hints: <guess>BEACH,N</guess>
### If you deliberately ignored hints to explore new letters: <guess>FLOWN,Y</guess>

### Response Format:

Think through the problem and feedback step by step. Make sure to first frame the rules based on given previous feedback. Ensure that the step by step thought process is within <think> </think> tags. Then, return your guessed word in the following format: <guess>GUESSED-WORD,Y/N</guess>.

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

# In[ ]:


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
def create_standard_dataset(n_samples, seed=42):
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

def create_difficult_dataset(n_puzzles, seed=42):
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

df_standard = create_standard_dataset(n_samples=int(0.2*n), seed=42)
df_difficult = create_difficult_dataset(n_puzzles=int(0.8* n), seed=42)

original_columns = ["prompt", "word_list", "past_guess_history", "secret"]
df_merged = pd.concat([df_standard[original_columns], df_difficult[original_columns]], ignore_index=True)

# Final deterministic shuffle
df_merged = df_merged.sample(frac=1, random_state=43).reset_index(drop=True)


# In[ ]:


df_merged


# In[ ]:


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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


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


# In[ ]:


def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    # THE FIX: Handle the empty state explicitly
    if not past_guesses:
        return "Make a new 5-letter word guess. This is the very first turn. There are no previous guesses. Do not assume any prior feedback."

    prompt = "Make a new 5-letter word guess.\n"
    prompt += "\nHere is the exact feedback from all previous guesses:\n"
    for i, past_guess in enumerate(past_guesses):
        prompt += f"Guess {i+1}: {past_guess}\n"
    return prompt


# import re

# def extract_guess(completion_text: str) -> str:
#     """Extracts the 5-letter guess, handling multiple tags, but STRICTLY enforces formatting."""
#     completion = str(completion_text)

#     # 1. Find ALL <guess> tags in the generated text ([\s\S] handles line breaks inside tags)
#     matches = re.findall(r"<guess>\s*([\s\S]*?)\s*</guess>", completion, re.IGNORECASE)

#     if matches:
#         # Work backwards from the last match found (the model's final decision is usually at the end)
#         for match in reversed(matches):
#             # Clean out any accidental punctuation, spaces, or symbols the model hallucinated
#             word = re.sub(r'[^A-Za-z]', '', match)
#             word = word.upper()

#             # If it's exactly 5 letters, we found our perfect target!
#             if len(word) == 5:
#                 return word

#         # If tags existed but none contained a valid 5-letter word, return the last one cleaned
#         # so the main loop can correctly log "Invalid Format: [Word]" and apply the -0.3 penalty.
#         return re.sub(r'[^A-Za-z]', '', matches[-1]).upper()

#     # 2. STRICT BENCHMARK ENFORCEMENT
#     # No fallback! If they forgot the tags entirely, they fail the format check.
#     return "MISSING_TAG"


# In[ ]:


import re

def extract_guess(completion_text: str) -> tuple[str, str]:
    """
    Extracts the 5-letter guess and the violation flag (Y/N).
    Returns: (guess_word, violation_flag)
    """
    completion = str(completion_text)

    # Regex to find <guess>WORD,FLAG</guess>
    matches = re.findall(r"<guess>\s*([\\s\\S]*?)\s*</guess>", completion, re.IGNORECASE)

    if matches:
        for match in reversed(matches):
            # Split by comma to separate word and flag
            parts = match.split(',')
            word_part = re.sub(r'[^A-Za-z]', '', parts[0]).upper()

            # Default flag to 'N' if not provided, otherwise clean the second part
            flag_part = 'N'
            if len(parts) > 1:
                flag_part = re.sub(r'[^A-Za-z]', '', parts[1]).upper()

            if len(word_part) == 5:
                return word_part, flag_part

        # Fallback for malformed content inside tags
        return re.sub(r'[^A-Za-z]', '', matches[-1].split(',')[0]).upper(), "INVALID"

    return "MISSING_TAG", "MISSING_TAG"


# In[ ]:


import re

def extract_guess_and_flag(completion_text: str) -> tuple[str, str]:
    """
    Extracts the 5-letter guess and the violation flag (Y/N).
    Returns: (guess_word, reported_flag)
    """
    completion = str(completion_text)

    # 1. Find ALL <guess> tags
    matches = re.findall(r"<guess>\s*([\s\S]*?)\s*</guess>", completion, re.IGNORECASE)

    if matches:
        # Work backwards from the last match
        for match in reversed(matches):
            # Split by comma (if present) to handle "WORD,N"
            parts = match.split(',')
            word_part = re.sub(r'[^A-Za-z]', '', parts[0]).upper()

            # Extract flag if exists, else default to 'N'
            flag_part = 'N'
            if len(parts) > 1:
                flag_part = re.sub(r'[^A-Za-z]', '', parts[1]).upper()
                flag_part = 'Y' if 'Y' in flag_part else 'N'

            if len(word_part) == 5:
                return word_part, flag_part

        # Fallback if tags exist but word is invalid length
        return re.sub(r'[^A-Za-z]', '', matches[-1].split(',')[0]).upper(), "INVALID"

    return "MISSING_TAG", "MISSING_TAG"


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

# In[ ]:


import math
import ast
import pandas as pd

def guess_value(prompts, completions, **kwargs) -> list[float]:
    """
    Revised Information Gain calculation logic.
    Integrates fixes for comma-separated extraction, stable entropy grouping, 
    and robust history handling.
    """

    # 1. Positional and character validation (unchanged)
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

    # 2. Filter lexicon based on prior turns
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

    # 3. Core Information Theory calculation
    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)

        # FIX A: If 1 or 0 candidates remain, no further information can be gained.
        if total_candidates <= 1:
            return 0.0, 0.0

        current_entropy = math.log2(total_candidates)
        feedback_groups = {}

        for word in candidates:
            # FIX B: Use the raw joined feedback string as the bucket key. 
            # This ensures that every unique feedback pattern is treated as a distinct group.
            feedback = validate_guess(word, guess, raw_feedback=True)
            feedback_pattern = "".join(feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)

        expected_entropy = 0
        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            # Expected entropy is the sum of (p * log2(group_size))
            expected_entropy += p * math.log2(group_size)

        expected_gain = current_entropy - expected_entropy
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0

        # Return normalized gain (max_gain set to 0.0 as it is unused in the main loop)
        return normalized_expected_gain, 0.0

    rewards = []
    word_lists = kwargs.get("word_list", [])
    past_guess_histories = kwargs.get("past_guess_history", [])
    cached_word_lists = {}

    for i in range(len(prompts)):
        try:
            comp = completions[i]
            completion_text = comp[0]["content"] if isinstance(comp, list) else str(comp)

            # FIX C: Call the new extractor and unpack the (guess, flag) tuple.
            # We only need the 'guess' word for the IG calculation.
            guess, _ = extract_guess_and_flag(completion_text)

            if guess == "MISSING_TAG" or len(guess) != 5:
                rewards.append(0.0)
                continue

            # Dictionary cache handling
            file_path = str(word_lists[i])
            if file_path not in cached_word_lists:
                lex_df = pd.read_csv(file_path)
                words_list = lex_df["Word"].astype(str).str.upper().tolist() 
                words_set = set(words_list)
                cached_word_lists[file_path] = (words_list, words_set)

            all_words_list, allowed_words_set = cached_word_lists[file_path]

            # Reject non-lexicon words
            if guess not in allowed_words_set:
                rewards.append(0.0)
                continue

            # FIX D: Handle history safely. 
            # If it's already a list (from the solve loop), use it directly; 
            # if it's a string (from a static dataset), parse it.
            history_raw = past_guess_histories[i]
            if isinstance(history_raw, str):
                past_guess_history = ast.literal_eval(history_raw)
            else:
                past_guess_history = history_raw

            normalized_expected_gain, _ = compute_normalized_information_gain(
                all_words_list,
                past_guess_history,
                guess
            )
            rewards.append(float(normalized_expected_gain))

        except Exception:
            # Catch-all to ensure the benchmark continues even if one row fails
            rewards.append(0.0)

    return rewards


# In[ ]:


# import pandas as pd
# import json
# import re
# import ast
# import math
# import kaggle_benchmarks as kbench
# from pathlib import Path

# # ==========================================
# # SETUP & DATA LOADING
# # ==========================================
# WORKDIR = Path(".")
# df = pd.read_json("wordle_eval_dataset1.jsonl", orient="records", lines=True)
# word_list_df = pd.read_csv("five_letter_words.csv") 
# allowed_words = word_list_df["Word"].str.upper().tolist()

# # ==========================================
# # HELPER FUNCTIONS
# # ==========================================

# def extract_guess_and_flag(completion_text: str) -> tuple[str, str]:
#     """Extracts the 5-letter guess and the violation flag (Y/N)."""
#     completion = str(completion_text)
#     matches = re.findall(r"<guess>\s*([\s\S]*?)\s*</guess>", completion, re.IGNORECASE)

#     if matches:
#         match = matches[-1]
#         parts = match.split(',')
#         word_part = re.sub(r'[^A-Za-z]', '', parts[0]).upper()

#         flag_part = 'N'
#         if len(parts) > 1:
#             flag_part = re.sub(r'[^A-Za-z]', '', parts[1]).upper()
#             flag_part = 'Y' if 'Y' in flag_part else 'N'

#         if len(word_part) == 5:
#             return word_part, flag_part
#         else:
#             word_match = re.search(r'[A-Z]{5}', word_part)
#             if word_match: return word_match.group(), flag_part

#     return "MISSING_TAG", "MISSING_TAG"

# def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
#     feedback = []
#     secret_list = list(secret.upper())
#     guess = guess.upper()
#     for i, (g_char, s_char) in enumerate(zip(guess, secret)):
#         if g_char == s_char:
#             feedback.append(f"{g_char}(✓) ")
#             secret_list[i] = None
#         else:
#             feedback.append(None)
#     for i, g_char in enumerate(guess):
#         if feedback[i] is None:
#             if g_char in secret_list:
#                 feedback[i] = f"{g_char}(-) "
#                 secret_list[secret_list.index(g_char)] = None
#             else:
#                 feedback[i] = f"{g_char}(x) "
#     if raw_feedback: return feedback
#     return "".join(feedback).strip()

# def count_constraint_violations(current_guess: str, past_history: list) -> int:
#     violations = 0
#     for past_word, past_fb in past_history:
#         if validate_guess(secret=current_guess, guess=past_word) != past_fb:
#             violations += 1
#     return violations

# def extract_think_block(completion_text: str) -> str:
#     match = re.search(r"<guess>", str(completion_text), re.IGNORECASE)
#     if match: return str(completion_text)[:match.start()].strip()
#     return str(completion_text).strip()

# # ==========================================
# # REWRITTEN GUESS_VALUE (SCORING)
# # ==========================================

# def guess_value(prompts, completions, **kwargs) -> list[float]:
#     def validate_internal(secret, guess):
#         """Standardized feedback: NO spaces for accurate filtering."""
#         secret, guess = secret.upper(), guess.upper()
#         feedback = [None] * 5
#         s_list = list(secret)
#         for i, (g, s) in enumerate(zip(guess, secret)):
#             if g == s:
#                 feedback[i] = f"{g}(✓)"; s_list[i] = None
#         for i, g in enumerate(guess):
#             if feedback[i] is None:
#                 if g in s_list:
#                     feedback[i] = f"{g}(-)"; s_list[s_list.index(g)] = None
#                 else: feedback[i] = f"{g}(x)"
#         return "".join(feedback)

#     def compute_ig(all_words, history, guess):
#         # Ensure history matches validate_internal (no spaces)
#         clean_history = [(h[0].upper(), h[1].replace(" ", "")) for h in history]

#         candidates = [
#             w for w in all_words 
#             if all(validate_internal(w, h[0]) == h[1] for h in clean_history)
#         ]

#         n = len(candidates)
#         if n <= 1: return 0.0

#         h_start = math.log2(n)
#         groups = {}
#         for w in candidates:
#             fb = validate_internal(w, guess)
#             groups.setdefault(fb, []).append(w)

#         h_end = sum((len(g)/n) * math.log2(len(g)) for g in groups.values())
#         return (h_start - h_end) / h_start

#     rewards = []
#     word_lists = kwargs.get("word_list", [])
#     past_histories = kwargs.get("past_guess_history", [])
#     cache = {}

#     for i in range(len(prompts)):
#         try:
#             comp = completions[i][0]["content"] if isinstance(completions[i], list) else str(completions[i])
#             guess, _ = extract_guess_and_flag(comp)

#             if guess == "MISSING_TAG" or len(guess) != 5:
#                 rewards.append(0.0); continue

#             path = str(word_lists[i])
#             if path not in cache:
#                 cache[path] = pd.read_csv(path)["Word"].str.upper().tolist()

#             h_raw = past_histories[i]
#             history = ast.literal_eval(h_raw) if isinstance(h_raw, str) else h_raw

#             rewards.append(float(compute_ig(cache[path], history, guess)))
#         except Exception:
#             rewards.append(0.0)
#     return rewards

# # ==========================================
# # 1. COGNITIVE FLEXIBILITY (SINGLE GAME)
# # ==========================================

# # ==========================================
# # 1. COGNITIVE FLEXIBILITY (SINGLE GAME)
# # ==========================================

# @kbench.task(name="cognitive_flexibility") 
# def cognitive_flexibility(llm, prompt: str, past_guess_history: list, secret: str, **kwargs) -> dict:
#     current_history = [(g[0], g[1]) for g in past_guess_history]
#     word_list_path = kwargs.get("word_list", "five_letter_words.csv")
#     current_prompt = prompt 
#     game_results = []

#     for _ in range(6 - len(current_history)):
#         curr_turn = len(current_history) + 1
#         state_tag = f"\n[SYSTEM STATE: Turn {curr_turn} of 6. {6 - len(current_history)} guesses left.]\n"

#         # --- CRITICAL FIX: Safe LLM Prompting ---
#         completion = None
#         try:
#             # Added a short sleep to help mitigate rate limits for Gemini
#             import time; time.sleep(2)
#             completion = llm.prompt(current_prompt + state_tag)
#         except Exception as e:
#             # If the API crashes (e.g. NoneType subscriptable, safety filter, timeout)
#             game_results.append({
#                 "secret": secret, "turn": curr_turn, "guess": "ERROR", 
#                 "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
#                 "error": f"API Exception: {str(e)[:50]}", "think_word_count": 0,
#                 "violations": 0, "reported_violation": "N"
#             })
#             break # Exit the game early on API failure

#         if not completion: 
#             game_results.append({
#                 "secret": secret, "turn": curr_turn, "guess": "ERROR", 
#                 "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
#                 "error": "API Exception: Empty Response (Blocked/Filtered)", "think_word_count": 0,
#                 "violations": 0, "reported_violation": "N"
#             })
#             break
#         # ----------------------------------------

#         think_text = extract_think_block(completion)
#         guess, flag = extract_guess_and_flag(completion)

#         if guess == "MISSING_TAG" or len(guess) != 5 or guess not in allowed_words:
#             game_results.append({
#                 "secret": secret, "turn": curr_turn, "guess": guess, 
#                 "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
#                 "error": f"Invalid Format: {guess}", "think_word_count": len(think_text.split()),
#                 "violations": 0, "reported_violation": flag
#             })
#             break 

#         v = count_constraint_violations(guess, current_history)
#         if v > 0:
#             info_gain = 0.0
#             turn_penalty = 0.0 if flag == "Y" else -0.3
#         else:
#             ig_list = guess_value([current_prompt], [completion], 
#                                   past_guess_history=[current_history], 
#                                   word_list=[word_list_path])
#             info_gain = ig_list[0] if ig_list else 0.0
#             turn_penalty = 0.0

#         is_win = guess == secret.upper()
#         entry = {
#             "secret": secret, "turn": curr_turn, "guess": guess, 
#             "info_gain": info_gain, "penalty": turn_penalty, "success_bonus": 1.0 if is_win else 0.0,
#             "violations": v, "reported_violation": flag, "think_word_count": len(think_text.split()), "error": "None"
#         }
#         game_results.append(entry)
#         if is_win: break

#         current_history.append((guess, validate_guess(secret, guess)))
#         current_prompt += f"\nAssistant: <guess>{guess},{flag}</guess>\nUser: Feedback: {validate_guess(secret, guess)}\n"

#     return {"game_turns": game_results}

# # @kbench.task(name="cognitive_flexibility") 
# # def cognitive_flexibility(llm, prompt: str, past_guess_history: list, secret: str, **kwargs) -> dict:
# #     current_history = [(g[0], g[1]) for g in past_guess_history]
# #     word_list_path = kwargs.get("word_list", "five_letter_words.csv")
# #     current_prompt = prompt 
# #     game_results = []

# #     for _ in range(6 - len(current_history)):
# #         curr_turn = len(current_history) + 1
# #         state_tag = f"\n[SYSTEM STATE: Turn {curr_turn} of 6. {6 - len(current_history)} guesses left.]\n"
# #         completion = llm.prompt(current_prompt + state_tag)
# #         if not completion: break

# #         think_text = extract_think_block(completion)
# #         guess, flag = extract_guess_and_flag(completion)

# #         if guess == "MISSING_TAG" or len(guess) != 5 or guess not in allowed_words:
# #             game_results.append({
# #                 "secret": secret, "turn": curr_turn, "guess": guess, 
# #                 "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
# #                 "error": f"Invalid Format: {guess}", "think_word_count": len(think_text.split())
# #             })
# #             break 

# #         v = count_constraint_violations(guess, current_history)
# #         if v > 0:
# #             info_gain = 0.0
# #             turn_penalty = 0.0 if flag == "Y" else -0.3
# #         else:
# #             ig_list = guess_value([current_prompt], [completion], 
# #                                   past_guess_history=[current_history], 
# #                                   word_list=[word_list_path])
# #             info_gain = ig_list[0] if ig_list else 0.0
# #             turn_penalty = 0.0

# #         is_win = guess == secret.upper()
# #         entry = {
# #             "secret": secret, "turn": curr_turn, "guess": guess, 
# #             "info_gain": info_gain, "penalty": turn_penalty, "success_bonus": 1.0 if is_win else 0.0,
# #             "violations": v, "reported_violation": flag, "think_word_count": len(think_text.split()), "error": "None"
# #         }
# #         game_results.append(entry)
# #         if is_win: break

# #         current_history.append((guess, validate_guess(secret, guess)))
# #         current_prompt += f"\nAssistant: <guess>{guess},{flag}</guess>\nUser: Feedback: {validate_guess(secret, guess)}\n"

# #     return {"game_turns": game_results}

# # ==========================================
# # 2. EVALUATION WRAPPER
# # ==========================================

# @kbench.task(name="evaluate_cognitive_flexibility")
# def evaluate_cognitive_flexibility(llm, df) -> float:
#     with kbench.client.enable_cache():
#         # Removed timeout parameter to fix the joblib warning
#         runs = cognitive_flexibility.evaluate(llm=[llm], evaluation_data=df, n_jobs=1, remove_run_files=True)

#     results_df = runs.as_dataframe()
#     if results_df.empty: return 0.0

#     combined = results_df.copy()

#     # Safely extract the dictionary results
#     combined['res'] = combined['result'].apply(lambda x: x.get('game_turns', []) if isinstance(x, dict) else [])
#     combined = combined.explode("res").reset_index(drop=True)

#     expanded = pd.json_normalize(combined["res"])
#     if expanded.empty: return 0.0

#     cols_to_drop = [c for c in expanded.columns if c in combined.columns]
#     combined = pd.concat([combined.drop(columns=["result", "res"] + cols_to_drop), expanded], axis=1)

#     error_ids = combined[combined['error'].str.contains('API Exception', na=False)]['id'].unique()
#     valid = combined[~combined['id'].isin(error_ids)].copy()

#     # --- DESIRED COLUMN ORDER ---
#     desired_order = [
#         'llm', 'prompt', 'past_guess_history', 'word_list', 'id', 
#         'secret', 'turn', 'guess', 'info_gain', 'penalty', 'success_bonus', 
#         'violations', 'reported_violation', 'think_word_count', 'error', 'game_score'
#     ]

#     if valid.empty: 
#         combined = combined.reindex(columns=desired_order + [c for c in combined.columns if c not in desired_order])
#         combined.to_csv(WORKDIR / "wordle_full_logs.csv", index=False, encoding='utf-8-sig')
#         return 0.0

#     # Aggregate Game Scores
#     game_grouped = valid.groupby('id').agg(
#         total_info_gain=('info_gain', 'sum'),
#         total_penalty=('penalty', 'sum'), 
#         turns=('turn', 'count'), 
#         success_bonus=('success_bonus', 'max'),
#         violations=('violations', 'sum')
#     )

#     game_grouped['avg_info_gain'] = game_grouped['total_info_gain'] / game_grouped['turns'].replace(0, 1)
#     game_grouped['game_score'] = game_grouped['avg_info_gain'] + game_grouped['total_penalty'] + game_grouped['success_bonus']

#     # --- INJECT GAME SCORE INTO LOGS ---
#     combined['game_score'] = ""
#     last_indices = combined.groupby('id').tail(1).index
#     combined.loc[last_indices, 'game_score'] = combined.loc[last_indices, 'id'].map(game_grouped['game_score'].round(4))

#     # --- ENFORCE ORIGINAL COLUMN ORDER BEFORE SAVING ---
#     combined = combined.reindex(columns=desired_order + [c for c in combined.columns if c not in desired_order])
#     combined.to_csv(WORKDIR / "wordle_full_logs.csv", index=False, encoding='utf-8-sig')

#     overall_score = float(game_grouped['game_score'].mean())

#     summary = {
#         "overall_benchmark_score": overall_score,
#         "win_rate": float(game_grouped['success_bonus'].mean()),
#         "avg_violations_per_game": float(game_grouped['violations'].mean()),
#         "api_error_rate": len(error_ids) / len(df)
#     }

#     (WORKDIR / "wordle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
#     print(json.dumps(summary, indent=2))

#     return overall_score


# In[ ]:


import pandas as pd
import json
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

def extract_guess_and_flag(completion_text: str) -> tuple[str, str]:
    """Extracts the 5-letter guess and the violation flag (Y/N)."""
    completion = str(completion_text)
    matches = re.findall(r"<guess>\s*([\s\S]*?)\s*</guess>", completion, re.IGNORECASE)

    if matches:
        match = matches[-1]
        parts = match.split(',')
        word_part = re.sub(r'[^A-Za-z]', '', parts[0]).upper()

        flag_part = 'N'
        if len(parts) > 1:
            flag_part = re.sub(r'[^A-Za-z]', '', parts[1]).upper()
            flag_part = 'Y' if 'Y' in flag_part else 'N'

        if len(word_part) == 5:
            return word_part, flag_part
        else:
            word_match = re.search(r'[A-Z]{5}', word_part)
            if word_match: return word_match.group(), flag_part

    return "MISSING_TAG", "MISSING_TAG"

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

def count_constraint_violations(current_guess: str, past_history: list) -> int:
    violations = 0
    for past_word, past_fb in past_history:
        if validate_guess(secret=current_guess, guess=past_word) != past_fb:
            violations += 1
    return violations

def extract_think_block(completion_text: str) -> str:
    match = re.search(r"<guess>", str(completion_text), re.IGNORECASE)
    if match: return str(completion_text)[:match.start()].strip()
    return str(completion_text).strip()

# ==========================================
# REWRITTEN GUESS_VALUE (SCORING)
# ==========================================

def guess_value(prompts, completions, **kwargs) -> list[float]:
    def validate_internal(secret, guess):
        """Standardized feedback: NO spaces for accurate filtering."""
        secret, guess = secret.upper(), guess.upper()
        feedback = [None] * 5
        s_list = list(secret)
        for i, (g, s) in enumerate(zip(guess, secret)):
            if g == s:
                feedback[i] = f"{g}(✓)"; s_list[i] = None
        for i, g in enumerate(guess):
            if feedback[i] is None:
                if g in s_list:
                    feedback[i] = f"{g}(-)"; s_list[s_list.index(g)] = None
                else: feedback[i] = f"{g}(x)"
        return "".join(feedback)

    def compute_ig(all_words, history, guess):
        # Ensure history matches validate_internal (no spaces)
        clean_history = [(h[0].upper(), h[1].replace(" ", "")) for h in history]

        candidates = [
            w for w in all_words 
            if all(validate_internal(w, h[0]) == h[1] for h in clean_history)
        ]

        n = len(candidates)
        if n <= 1: return 0.0

        h_start = math.log2(n)
        groups = {}
        for w in candidates:
            fb = validate_internal(w, guess)
            groups.setdefault(fb, []).append(w)

        h_end = sum((len(g)/n) * math.log2(len(g)) for g in groups.values())
        return (h_start - h_end) / h_start

    rewards = []
    word_lists = kwargs.get("word_list", [])
    past_histories = kwargs.get("past_guess_history", [])
    cache = {}

    for i in range(len(prompts)):
        try:
            comp = completions[i][0]["content"] if isinstance(completions[i], list) else str(completions[i])
            guess, _ = extract_guess_and_flag(comp)

            if guess == "MISSING_TAG" or len(guess) != 5:
                rewards.append(0.0); continue

            path = str(word_lists[i])
            if path not in cache:
                cache[path] = pd.read_csv(path)["Word"].str.upper().tolist()

            h_raw = past_histories[i]
            history = ast.literal_eval(h_raw) if isinstance(h_raw, str) else h_raw

            rewards.append(float(compute_ig(cache[path], history, guess)))
        except Exception:
            rewards.append(0.0)
    return rewards

# ==========================================
# 1. COGNITIVE FLEXIBILITY (SINGLE GAME)
# ==========================================

# ==========================================
# 1. COGNITIVE FLEXIBILITY (SINGLE GAME)
# ==========================================

@kbench.task(name="cognitive_flexibility") 
def cognitive_flexibility(llm, prompt: str, past_guess_history: list, secret: str, **kwargs) -> dict:
    current_history = [(g[0], g[1]) for g in past_guess_history]
    word_list_path = kwargs.get("word_list", "five_letter_words.csv")
    current_prompt = prompt 
    game_results = []

    for _ in range(6 - len(current_history)):
        curr_turn = len(current_history) + 1
        state_tag = f"\n[SYSTEM STATE: Turn {curr_turn} of 6. {6 - len(current_history)} guesses left.]\n"

        # --- CRITICAL FIX: Safe LLM Prompting ---
        completion = None
        try:
            # Added a short sleep to help mitigate rate limits for Gemini
            # import time
            # time.sleep(2)
            completion = llm.prompt(current_prompt + state_tag)
        except Exception as e:
            # If the API crashes (e.g. NoneType subscriptable, safety filter, timeout)
            game_results.append({
                "secret": secret, "turn": curr_turn, "guess": "ERROR", 
                "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
                "error": f"API Exception: {str(e)[:50]}", "think_word_count": 0,
                "violations": 0, "reported_violation": "N"
            })
            break # Exit the game early on API failure

        if not completion: 
            game_results.append({
                "secret": secret, "turn": curr_turn, "guess": "ERROR", 
                "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
                "error": "API Exception: Empty Response (Blocked/Filtered)", "think_word_count": 0,
                "violations": 0, "reported_violation": "N"
            })
            break
        # ----------------------------------------

        think_text = extract_think_block(completion)
        guess, flag = extract_guess_and_flag(completion)

        if guess == "MISSING_TAG" or len(guess) != 5 or guess not in allowed_words:
            game_results.append({
                "secret": secret, "turn": curr_turn, "guess": guess, 
                "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
                "error": f"Invalid Format: {guess}", "think_word_count": len(think_text.split()),
                "violations": 0, "reported_violation": flag
            })
            break 

        v = count_constraint_violations(guess, current_history)

        ig_list = guess_value([current_prompt], [completion], 
                              past_guess_history=[current_history], 
                              word_list=[word_list_path])
        calculated_ig = ig_list[0] if ig_list else 0.0

        if v > 0:
            if flag == "Y":
                info_gain = calculated_ig
                turn_penalty = 0.0
            else:
                info_gain = 0.0
                turn_penalty = -0.3
        else:
            info_gain = calculated_ig
            turn_penalty = 0.0

        is_win = guess == secret.upper()
        entry = {
            "secret": secret, "turn": curr_turn, "guess": guess, 
            "info_gain": info_gain, "penalty": turn_penalty, "success_bonus": 1.0 if is_win else 0.0,
            "violations": v, "reported_violation": flag, "think_word_count": len(think_text.split()), "error": "None"
        }
        game_results.append(entry)
        if is_win: break

        current_history.append((guess, validate_guess(secret, guess)))
        current_prompt += f"\nAssistant: <guess>{guess},{flag}</guess>\nUser: Feedback: {validate_guess(secret, guess)}\n"

    return {"game_turns": game_results}

# @kbench.task(name="cognitive_flexibility") 
# def cognitive_flexibility(llm, prompt: str, past_guess_history: list, secret: str, **kwargs) -> dict:
#     current_history = [(g[0], g[1]) for g in past_guess_history]
#     word_list_path = kwargs.get("word_list", "five_letter_words.csv")
#     current_prompt = prompt 
#     game_results = []

#     for _ in range(6 - len(current_history)):
#         curr_turn = len(current_history) + 1
#         state_tag = f"\n[SYSTEM STATE: Turn {curr_turn} of 6. {6 - len(current_history)} guesses left.]\n"
#         completion = llm.prompt(current_prompt + state_tag)
#         if not completion: break

#         think_text = extract_think_block(completion)
#         guess, flag = extract_guess_and_flag(completion)

#         if guess == "MISSING_TAG" or len(guess) != 5 or guess not in allowed_words:
#             game_results.append({
#                 "secret": secret, "turn": curr_turn, "guess": guess, 
#                 "info_gain": 0.0, "penalty": 0.0, "success_bonus": 0.0,
#                 "error": f"Invalid Format: {guess}", "think_word_count": len(think_text.split())
#             })
#             break 

#         v = count_constraint_violations(guess, current_history)
#         
#         ig_list = guess_value([current_prompt], [completion], 
#                               past_guess_history=[current_history], 
#                               word_list=[word_list_path])
#         calculated_ig = ig_list[0] if ig_list else 0.0
#
#         if v > 0:
#             if flag == "Y":
#                 info_gain = calculated_ig
#                 turn_penalty = 0.0
#             else:
#                 info_gain = 0.0
#                 turn_penalty = -0.3
#         else:
#             info_gain = calculated_ig
#             turn_penalty = 0.0

#         is_win = guess == secret.upper()
#         entry = {
#             "secret": secret, "turn": curr_turn, "guess": guess, 
#             "info_gain": info_gain, "penalty": turn_penalty, "success_bonus": 1.0 if is_win else 0.0,
#             "violations": v, "reported_violation": flag, "think_word_count": len(think_text.split()), "error": "None"
#         }
#         game_results.append(entry)
#         if is_win: break

#         current_history.append((guess, validate_guess(secret, guess)))
#         current_prompt += f"\nAssistant: <guess>{guess},{flag}</guess>\nUser: Feedback: {validate_guess(secret, guess)}\n"

#     return {"game_turns": game_results}

# ==========================================
# 2. EVALUATION WRAPPER
# ==========================================

@kbench.task(name="evaluate_cognitive_flexibility")
def evaluate_cognitive_flexibility(llm, df) -> float:
    with kbench.client.enable_cache():
        # Removed timeout parameter to fix the joblib warning
        runs = cognitive_flexibility.evaluate(llm=[llm], evaluation_data=df, n_jobs=1, remove_run_files=True)

    results_df = runs.as_dataframe()
    if results_df.empty: return 0.0

    combined = results_df.copy()

    # Safely extract the dictionary results
    combined['res'] = combined['result'].apply(lambda x: x.get('game_turns', []) if isinstance(x, dict) else [])
    combined = combined.explode("res").reset_index(drop=True)

    expanded = pd.json_normalize(combined["res"])
    if expanded.empty: return 0.0

    cols_to_drop = [c for c in expanded.columns if c in combined.columns]
    combined = pd.concat([combined.drop(columns=["result", "res"] + cols_to_drop), expanded], axis=1)

    error_ids = combined[combined['error'].str.contains('API Exception', na=False)]['id'].unique()
    valid = combined[~combined['id'].isin(error_ids)].copy()

    # --- DESIRED COLUMN ORDER ---
    desired_order = [
        'llm', 'prompt', 'past_guess_history', 'word_list', 'id', 
        'secret', 'turn', 'guess', 'info_gain', 'penalty', 'success_bonus', 
        'violations', 'reported_violation', 'think_word_count', 'error', 'game_score'
    ]

    if valid.empty: 
        combined = combined.reindex(columns=desired_order + [c for c in combined.columns if c not in desired_order])
        combined.to_csv(WORKDIR / "wordle_full_logs.csv", index=False, encoding='utf-8-sig')
        return 0.0

    # Aggregate Game Scores
    game_grouped = valid.groupby('id').agg(
        total_info_gain=('info_gain', 'sum'),
        total_penalty=('penalty', 'sum'), 
        turns=('turn', 'count'), 
        success_bonus=('success_bonus', 'max'),
        violations=('violations', 'sum')
    )

    game_grouped['avg_info_gain'] = game_grouped['total_info_gain'] / game_grouped['turns'].replace(0, 1)
    game_grouped['game_score'] = game_grouped['avg_info_gain'] + game_grouped['total_penalty'] + game_grouped['success_bonus']

    # --- INJECT GAME SCORE INTO LOGS ---
    combined['game_score'] = ""
    last_indices = combined.groupby('id').tail(1).index
    combined.loc[last_indices, 'game_score'] = combined.loc[last_indices, 'id'].map(game_grouped['game_score'].round(4))

    # --- ENFORCE ORIGINAL COLUMN ORDER BEFORE SAVING ---
    combined = combined.reindex(columns=desired_order + [c for c in combined.columns if c not in desired_order])
    combined.to_csv(WORKDIR / "wordle_full_logs.csv", index=False, encoding='utf-8-sig')

    overall_score = float(game_grouped['game_score'].mean())

    summary = {
        "overall_benchmark_score": overall_score,
        "win_rate": float(game_grouped['success_bonus'].mean()),
        "avg_violations_per_game": float(game_grouped['violations'].mean()),
        "api_error_rate": len(error_ids) / len(df)
    }

    (WORKDIR / "wordle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    return overall_score


# In[ ]:


# kbench.llms
# llm=kbench.llms["openai/gpt-5.4-2026-03-05"]
# llm = kbench.llms['openai/gpt-5.4-mini-2026-03-17']
# llm = kbench.llms['google/gemma-4-31b']


# In[ ]:


evaluate_cognitive_flexibility.run(kbench.llm,df)


# In[ ]:


get_ipython().run_line_magic('choose', 'evaluate_cognitive_flexibility')


# In[ ]:


# import zipfile
# import os

# # Name of the output zip file
# zip_name = "archive.zip"

# with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     for file in os.listdir('.'):
#         # 1. Skip the zip file itself so it doesn't try to include itself
#         # 2. Skip directories if you only want files
#         if file != zip_name and os.path.isfile(file):
#             zipf.write(file)
#             print(f"Zipped: {file}")

# print(f"Done! Created {zip_name}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




