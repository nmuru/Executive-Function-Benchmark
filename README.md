# Executive Function Benchmark for AI Models

A benchmark suite for evaluating Executive Function capabilities of AI models using Wordle game and Information Gain optimization

Unlike traditional "plain-vanilla" win-rate benchmarks, this framework measures the working memory drift, cognitive efficiency, and constraint-adherence of AI agents under dynamic restrictions.

It is based on the kaggle benchmarks and allows cross-task insights showing the potential of wordle game as an efficient benchmark for Executive Function of AI Models

## 📊 Core Cognitive Pillars Evaluated
* **Working Memory:** Context retention and rule-drift resistance over multi-turn interactions.
* **Cognitive Flexibility:** Adaptive reasoning when shifting strategies dynamically.
* **Inhibitory Control:** Impulse suppression and mistake minimization under stress.

## 🚀 Live Interactive Dashboard
Explore the full, interactive cross-task leaderboard, adjust scoring weights dynamically, and view normalized model cognitive maps here:
👉 **[Live Dashboard](https://nmuru.github.io/Executive-Function-Benchmark/)**

## 📂 Repository Structure
* `/data` - Standardized JSON evaluation outputs for single-turn, multi-turn, and cognitive flexibility tasks.
* `/templates` - HTML UI layout templates.
* `generate.py` - Core Python pipeline for processing Kaggle leaderboard metrics and generating the static site.
  

---
* Data Source: All metrics are extracted from standardized benchmark notebook runs on the <a href="https://www.kaggle.com/benchmarks/murugesann/executivefunction-infogain-wordle-bench" target="_blank" class="underline font-semibold hover:text-blue-900">
Kaggle Benchmarks Platform </a>  
