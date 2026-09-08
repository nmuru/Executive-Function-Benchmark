from pathlib import Path
import pandas as pd
import numpy as np
import html as html_lib
import re

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR_5L = BASE_DIR / "Wordle-5L-Dashboard" / "data"
DATA_DIR_6L = BASE_DIR / "Wordle-6L-Dashboard" / "data"
OUTPUT_FILE = BASE_DIR / "index.html"

FILE_5L = DATA_DIR_5L / "murugesann_executivefunction-infogain-wordle-bench_leaderboard.csv"
FILE_6L = DATA_DIR_6L / "murugesann_wordle-benchmark_leaderboard.csv"
FILE_EXTERNAL = DATA_DIR / "benchmark_data.xlsx"
FILE_SOLVER = DATA_DIR / "Deterministic_solver_scores.txt"
def find_resource_csv(data_dir):
    candidates = [
        data_dir / "resource_metrics" / "resource_metrics.csv",
        data_dir / "resource_metrics.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Also tolerate a differently named CSV inside the resource_metrics directory.
    resource_dir = data_dir / "resource_metrics"
    if resource_dir.exists():
        csvs = sorted(resource_dir.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
    return candidates[0]

RESOURCE_COMBINED = BASE_DIR / "resource_metrics_combined.csv"

# ---------------------------------------------------------------------------
# The parent page is intentionally embedded here.
# generate_parent.py creates index.html afresh and does NOT read an existing
# index.html as a template.
# ---------------------------------------------------------------------------

TEMPLATE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Executive Function Benchmark — Prototype</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  .table-wrap{overflow-x:auto}
  th{white-space:nowrap}
  td,th{font-variant-numeric:tabular-nums}
  .rank-1{background:#eff6ff;font-weight:800}
  .score{font-weight:800;color:#111827}
</style>
</head>
<body class="bg-slate-50 text-slate-900">
<main class="max-w-7xl mx-auto px-5 py-10 md:px-8 md:py-14">
<header class="max-w-4xl">
  <div class="text-xs font-bold uppercase tracking-[0.18em] text-blue-600">AI Evaluation Framework</div>
  <h1 class="mt-3 text-4xl md:text-5xl font-bold tracking-tight">Executive Function Benchmark</h1>
  <p class="mt-4 text-lg text-slate-600 leading-relaxed">A combined 5-letter + 6-letter Wordle benchmark, presented alongside selected SOTA benchmarks for the same six models.</p>
  <div class="mt-6 h-1 w-16 rounded-full bg-blue-600"></div>
</header>

<section class="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
  <a href="Wordle-5L-Dashboard/index.html" class="group block rounded-2xl border-2 border-blue-200 bg-blue-50 p-7 shadow-sm hover:shadow-md hover:border-blue-400 transition">
    <div class="text-xs font-bold uppercase tracking-[0.16em] text-blue-600">5-Letter Wordle</div>
    <div class="mt-2 text-2xl font-bold text-slate-900">Explore the 5L Benchmark</div>
    <p class="mt-2 text-sm text-slate-600">View the full 5-letter Wordle benchmark, task results, rankings, and model performance.</p>
    <div class="mt-5 text-sm font-bold text-blue-700 group-hover:underline">Open 5L Dashboard →</div>
  </a>
  <a href="Wordle-6L-Dashboard/index.html" class="group block rounded-2xl border-2 border-blue-200 bg-blue-50 p-7 shadow-sm hover:shadow-md hover:border-blue-400 transition">
    <div class="text-xs font-bold uppercase tracking-[0.16em] text-blue-600">6-Letter Wordle</div>
    <div class="mt-2 text-2xl font-bold text-slate-900">Explore the 6L Benchmark</div>
    <p class="mt-2 text-sm text-slate-600">View the full 6-letter Wordle benchmark, task results, rankings, and model performance.</p>
    <div class="mt-5 text-sm font-bold text-blue-700 group-hover:underline">Open 6L Dashboard →</div>
  </a>
</section>


<section class="mt-12">
  <div class="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
    <div>
      <div class="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Primary ranking</div>
      <h2 class="mt-1 text-2xl font-bold">Combined 5L + 6L Ranking</h2>
      <p class="mt-2 text-slate-600 max-w-3xl">The ranking below is calculated from models with complete required task results in both the 5L and 6L Wordle leaderboard files. The two benchmark scores are combined into a single Overall Score.</p>
    </div>
    <span class="inline-flex self-start rounded-full bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700">{{MODEL_COUNT}} models</span>
  </div>

  <div class="mt-6 bg-white border border-slate-200 rounded-2xl shadow-sm table-wrap">
    <table class="w-full text-sm">
      <thead class="bg-slate-50 text-xs uppercase tracking-wider text-slate-500">
        <tr>
          <th class="px-5 py-4 text-left">Rank</th><th class="px-5 py-4 text-left">Model</th><th class="px-5 py-4 text-right">Overall</th><th class="px-5 py-4 text-right">Single-Turn</th><th class="px-5 py-4 text-right">Multi-Turn</th><th class="px-5 py-4 text-right">Cognitive Flex.</th>
        </tr>
      </thead>
      <tbody id="combined-body" class="divide-y divide-slate-100">{{COMBINED_ROWS}}</tbody>
    </table>
  </div>
  <p class="mt-3 text-xs text-slate-500">Models are included automatically when all required 5L and 6L task results are present. The Overall Score is the equal-weight average of the weighted 5L and 6L benchmark scores.</p>
</section>

<section class="mt-14">
  <div>
    <div class="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Operational highlights</div>
    <h2 class="mt-1 text-2xl font-bold">Operational Highlights</h2>
    <p class="mt-2 text-slate-600 max-w-4xl">The strongest combined 5L + 6L results on cost efficiency, decision velocity, and information density. These metrics use the resource data from both benchmark suites and include only models with complete resource coverage in both resource files.</p>
  </div>
  {{OPERATIONAL_HIGHLIGHTS}}
</section>

<section class="mt-14">
  <div>
    <div class="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Resource footprint</div>
    <h2 class="mt-1 text-2xl font-bold">Resource &amp; Efficiency Metrics</h2>
    <p class="mt-2 text-slate-600 max-w-4xl">Combined resource footprint across the 5L and 6L resource files. The Combined Efficiency Score gives equal weight to bang-for-buck, velocity, and information density after max-value normalization within the eligible model set.</p>
  </div>
  {{RESOURCE_EFFICIENCY}}
</section>

<section class="mt-14">
  <div>
    <div class="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">External context</div>
    <h2 class="mt-1 text-2xl font-bold">SOTA Benchmark Comparison</h2>
    <p class="mt-2 text-slate-600 max-w-4xl">Wordle scores are shown alongside selected external benchmark scores for the same models. A rank in parentheses is calculated only within these six models. N/A means no sufficiently clear public score was found for this prototype. Scores are not recomputed or normalized across benchmarks.</p>
  </div>

  <div class="mt-6 bg-white border border-slate-200 rounded-2xl shadow-sm table-wrap">
    
<table class="w-full text-left border-collapse min-w-[1100px]">
  <thead>
    <tr class="bg-gray-100 text-xs text-gray-600 font-semibold uppercase tracking-wider">
      <th rowspan="2" class="py-3 px-4 border-b">Model</th>
      <th colspan="2" class="py-3 px-4 border-b text-center bg-blue-50 text-blue-900">Wordle Combined</th>
      <th colspan="2" class="py-3 px-4 border-b text-center">Artificial Analysis Index</th>
      <th colspan="2" class="py-3 px-4 border-b text-center">GPQA Diamond</th>
      <th colspan="2" class="py-3 px-4 border-b text-center">MMLU-Pro</th>
      <th colspan="2" class="py-3 px-4 border-b text-center">Humanity's Last Exam</th>
    </tr>
    <tr class="bg-gray-50 text-xs text-gray-500">
      <th class="py-2 px-3 border-b text-center bg-blue-50">Score</th>
      <th class="py-2 px-3 border-b text-center bg-blue-50">Rank</th>
      <th class="py-2 px-3 border-b text-center">Score</th>
      <th class="py-2 px-3 border-b text-center">Rank</th>
      <th class="py-2 px-3 border-b text-center">Score</th>
      <th class="py-2 px-3 border-b text-center">Rank</th>
      <th class="py-2 px-3 border-b text-center">Score</th>
      <th class="py-2 px-3 border-b text-center">Rank</th>
      <th class="py-2 px-3 border-b text-center">Score</th>
      <th class="py-2 px-3 border-b text-center">Rank</th>
    </tr>
  </thead>
  <tbody>
{{EXTERNAL_ROWS}}
  </tbody>
</table>

  </div>
  <div class="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900 leading-relaxed">
    External benchmark scores are shown as reported. Rank is calculated only within the six models shown on this page.
  </div>
</section>


<section class="mt-14">
  <div class="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Notes</div>
  <h2 class="mt-1 text-2xl font-bold">Notes</h2>
  <div class="mt-5 bg-white border border-slate-200 rounded-2xl p-6 text-sm text-slate-600 leading-relaxed">
    <ul class="list-disc pl-5 space-y-4">

	<li>At present, GPT-5.6-Sol is not available for benchmarking in kaggle. Similarly, Grok 4.6 could not be benchmarked. </li> 

      <li><sup>*</sup> The deterministic solver score is the maximum possible score when the Wordle game is solved by a Python program using the entire word list and search tools.</li>

      <li>The deterministic solver feature applies only to 6L Wordle and has not been performed for 5L Wordle.</li>

      <li>Wordle game scores are scaled against the deterministic solver score. Thus, if a solver-normalized score is 85%, it represents a 15% gap relative to the maximum possible score.</li>

      <li><strong>6L success-bonus tiering:</strong> Compared with 5-letter Wordle, the 6-letter benchmark uses a tiered success reward based on the turn in which the model identifies the correct secret word. Earlier success receives a higher reward, while success on later turns receives a progressively lower reward. A model that finds the correct word only on the sixth and final turn therefore receives the lowest success reward for a successful game.</li>

      <li>The current Multi-turn kaggle notebook (unlike cognitive flexibility task) does not do success bonus calculation based on turn in which model succeeds. We have now extracted data from multi turn kaggle task benchmark scores and then manually recalculated and scaled it to reference deterministic solver score. Hence, the scores for Multi-turn in this dashboard will not match with scores available against the benchmark in kaggle. Since scores are normalized with deterministic solver score, the scores are now expressed in terms of percentages. </li> 

      <li><strong>Single-Turn coverage and task weights:</strong> For 5-letter Wordle, Single-Turn uses only <code>evaluate_wordle_single_turn_v2</code>, which covers 200 rows (200 API calls), while the 6-letter Single-Turn task covers 40 rows. The 5L task weights are 20% Single-Turn, 50% Multi-Turn and 30% Cognitive Flexibility. The 6L task weights are 10% Single-Turn, 50% Multi-Turn and 40% Cognitive Flexibility.</li>

      <li>The overall Wordle score is the equal-weight average of the resulting weighted 5L and 6L benchmark scores.</li>

      <li>The front-page model set is discovered automatically from the current leaderboard CSVs. A model is included only when all required Wordle tasks are present in both the 5L and 6L files. Model versions are not substituted for one another, so Claude Opus 4.8 and Claude Opus 5 are treated as separate models.</li>
    </ul>
  </div>
</section>


<footer class="mt-12 pt-6 border-t border-slate-200 text-xs text-slate-400">Executive Function Benchmark · Combined 5L + 6L ranking with selected external benchmark context</footer>
</main>


</body>
</html>
"""

def canonical_model_id(value):
    """Conservative key for matching the same model across 5L and 6L CSVs."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-default$", "", value)
    return value


def display_model_name(model_id):
    """Readable label. Unknown future models are included automatically."""
    if model_id is None:
        return None

    known = {
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
        "gemini-3-flash-preview": "Gemini 3.6 Flash",
        "gemini-3.6-flash": "Gemini 3.6 Flash",
        "gpt-5.6-sol": "GPT-5.6 Sol",
	"gpt-5.6-terra": "GPT-5.6 Terra", 
        "gemma-4-31b-it": "Gemma 4 31B",
        "grok-4.20-0309-reasoning": "Grok 4.20 Reasoning",
        "claude-opus-4-8": "Claude Opus 4.8",
        "claude-opus-5": "Claude Opus 5",
    }
    return known.get(model_id, model_id.replace("-", " ").title())


def normalize_model(value):
    return display_model_name(canonical_model_id(value))


def discover_common_models(path_5l, path_6l, task_names_5l, task_names_6l):
    """Return model IDs with valid numeric results for every required task in BOTH files.

    A task name appearing in a CSV is not sufficient. Its Numerical_Result must
    contain at least one numeric value. This prevents partially populated
    leaderboard rows from entering the combined ranking.
    """
    df5 = pd.read_csv(path_5l)
    df6 = pd.read_csv(path_6l)

    def complete_models(df, required_tasks):
        found = set()
        for raw_model, group in df.groupby("Model", dropna=True):
            model_id = canonical_model_id(raw_model)
            complete = True

            for task in required_tasks:
                values = pd.to_numeric(
                    group.loc[group["Task_Name"] == task, "Numerical_Result"],
                    errors="coerce",
                )
                if values.dropna().empty:
                    complete = False
                    break

            if complete:
                found.add(model_id)

        return found

    common = sorted(
        complete_models(df5, set(task_names_5l.values()))
        & complete_models(df6, set(task_names_6l.values()))
    )

    if not common:
        raise ValueError(
            "No model has valid numeric results for all required 5L and 6L tasks."
        )

    print("Common complete models:")
    for model_id in common:
        print(f"  {model_id} -> {display_model_name(model_id)}")

    return common


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")


def load_task_table(path, task_names, weights):
    """Load one leaderboard into a canonical model-id table."""
    df = pd.read_csv(path)

    required = {"Model", "Task_Name", "Numerical_Result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns: {sorted(missing)}"
        )

    relevant = df[df["Task_Name"].isin(task_names.values())].copy()
    print(
        f"{path.name} tasks found: "
        f"{sorted(relevant['Task_Name'].dropna().astype(str).unique().tolist())}"
    )

    rows = []
    for raw_model, group in relevant.groupby("Model", dropna=True):
        model_id = canonical_model_id(raw_model)

        vals = {}
        complete = True
        for key, task in task_names.items():
            values = pd.to_numeric(
                group.loc[group["Task_Name"] == task, "Numerical_Result"],
                errors="coerce",
            ).dropna()

            if values.empty:
                complete = False
                break

            vals[key.upper()] = float(values.mean())

        if not complete:
            continue

        score = sum(vals[k] * weights[k] for k in ["ST", "MT", "CF"])
        rows.append({
            "model_id": model_id,
            "model": display_model_name(model_id),
            "ST": vals["ST"],
            "MT": vals["MT"],
            "CF": vals["CF"],
            "score": score,
        })

    return pd.DataFrame(rows)


def load_wordle_scores():
    tasks_5l = {
        "st": "evaluate_wordle_single_turn_v2",
        "mt": "evaluate_wordle_multi_turn",
        "cf": "evaluate_cognitive_flexibility",
    }
    tasks_6l = {
        "st": "evaluate_wordle_6L_single_turn_larger",
        "mt": "evaluate_6Lwordle_multi_turn",
        "cf": "evaluate_6Lwordle_cognitive_flexibility",
    }

    # Your chosen benchmark-specific weights.
    weights_5l = {"ST": 0.20, "MT": 0.50, "CF": 0.30}
    weights_6l = {"ST": 0.20, "MT": 0.50, "CF": 0.30}

    scores_5l = load_task_table(FILE_5L, tasks_5l, weights_5l).rename(columns={
        "ST": "st_5l", "MT": "mt_5l", "CF": "cf_5l",
        "score": "score_5l",
    })
    scores_6l = load_task_table(FILE_6L, tasks_6l, weights_6l).rename(columns={
        "ST": "st_6l", "MT": "mt_6l", "CF": "cf_6l",
        "score": "score_6l",
    })

    # Only models with complete numeric data in BOTH leaderboards are eligible.
    common_ids = sorted(
        set(scores_5l["model_id"]) & set(scores_6l["model_id"])
    )

    if not common_ids:
        raise ValueError(
            "No model has complete numeric results for all required tasks "
            "in both 5L and 6L leaderboards."
        )

    print("Common complete models:")
    for model_id in common_ids:
        print(f"  {model_id} -> {display_model_name(model_id)}")

    a = scores_5l[scores_5l["model_id"].isin(common_ids)].copy()
    b = scores_6l[scores_6l["model_id"].isin(common_ids)].copy()

    # Defensive duplicate check. There should be exactly one complete row per
    # canonical model in each benchmark.
    if a["model_id"].duplicated().any():
        raise ValueError("Duplicate complete 5L model IDs detected.")
    if b["model_id"].duplicated().any():
        raise ValueError("Duplicate complete 6L model IDs detected.")

    df = pd.DataFrame({"model_id": common_ids})
    df = df.merge(
        a.drop(columns=["model"]),
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    df = df.merge(
        b.drop(columns=["model"]),
        on="model_id",
        how="inner",
        validate="one_to_one",
    )

    df["model"] = df["model_id"].map(display_model_name)
    df["overall"] = (df["score_5l"] + df["score_6l"]) / 2

    # No missing values are allowed into the ranking.
    numeric_cols = [
        "score_5l", "score_6l", "st_5l", "mt_5l", "cf_5l",
        "st_6l", "mt_6l", "cf_6l", "overall",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)

    if df.empty:
        raise ValueError("All common models became incomplete after merging 5L and 6L data.")

    df["rank"] = df["overall"].rank(method="min", ascending=False).astype("Int64")

    return df.sort_values(["rank", "model_id"]).reset_index(drop=True)


def load_external_benchmarks():
    """
    Read benchmark_data.xlsx.

    The workbook's Sheet1 has a fixed two-row layout:

        Row 1: Model | GPQA Diamond | blank | MMLU | blank | HLE | blank | MMU-PRO | blank
        Row 2:       | Score        | Rank  | Score | Rank  | Score | Rank | Score  | Rank
        Row 3+: model data

    Wordle data is intentionally ignored. Three external benchmark score columns
    are read from the workbook. The Artificial Analysis Index is supplied as
    hard-coded values. Ranks are recalculated across the selected models.
    """

    raw = pd.read_excel(
        FILE_EXTERNAL,
        sheet_name="Sheet1",
        header=None,
    )

    # Fixed columns in the supplied workbook:
    # A = Model
    # B = GPQA score
    # D = MMLU score
    # F = HLE score
    col_map = {
        "GPQA Diamond": 1,
        "MMLU-Pro": 3,
        "HLE": 5,
    }

    # Artificial Analysis Intelligence Index values supplied manually.
    artificial_analysis_index = {
        "Gemini 3.1 Pro Preview": 48,
        "GPT-5.6 Sol": 56,
        "Gemini 3.6 Flash": 52,
        "Grok 4.20 Reasoning": 37,
        "Gemma 4 31B": 30,
        "Claude Opus 5": 45,
        "GPT-5.6 Terra": 33,
        "Gemini 3.8 Flash": 40,
    }

    # Start with every Wordle-selected model so hard-coded AA Index values
    # are preserved even if the model is not present in the Excel workbook.
    records_by_model = {
        model: {
            "model": model,
            "Artificial Analysis Index": artificial_analysis_index.get(model, np.nan),
            **{benchmark: np.nan for benchmark in col_map},
        }
        for model in DISPLAY_MODELS_DYNAMIC
    }

    # Overlay GPQA/MMLU-Pro/HLE values from Excel where a matching model exists.
    for i in range(2, len(raw)):
        model_value = raw.iloc[i, 0]

        if pd.isna(model_value):
            continue

        model = normalize_model(model_value)

        if model not in records_by_model:
            continue

        for benchmark, col in col_map.items():
            value = pd.to_numeric(
                raw.iloc[i, col],
                errors="coerce",
            )
            records_by_model[model][benchmark] = value

    df = pd.DataFrame(list(records_by_model.values()))

    # Rank only among the dynamically selected Wordle models.
    for benchmark in ["Artificial Analysis Index", *col_map]:
        df[f"{benchmark}_rank"] = (
            pd.to_numeric(
                df[benchmark],
                errors="coerce",
            )
            .rank(
                method="min",
                ascending=False,
            )
            .astype("Int64")
        )

    print("External benchmark rows loaded:", len(df))
    print(
        df[
            [
                "model",
                "Artificial Analysis Index",
                "GPQA Diamond",
                "MMLU-Pro",
                "HLE",
            ]
        ].to_string(index=False)
    )

    return df


def load_deterministic_solver_score():
    """
    Read the deterministic solver score from data/Deterministic_solver_scores.txt.
    The source may contain different solver entries; the solver baseline is independent of model selection.
    """
    raw = FILE_SOLVER.read_text(encoding="utf-8")
    matches = re.findall(
        r"(?im)^\s*(5-Letter Wordle|6-Letter Wordle)\s*-\s*([0-9.]+)\s*$",
        raw,
    )
    scores = {label: float(value) for label, value in matches}

    if "5-Letter Wordle" not in scores or "6-Letter Wordle" not in scores:
        raise ValueError(
            f"{FILE_SOLVER.name}: expected both 5-Letter Wordle and 6-Letter Wordle scores"
        )

    if abs(scores["5-Letter Wordle"] - scores["6-Letter Wordle"]) > 1e-12:
        raise ValueError(
            f"{FILE_SOLVER.name}: 5L and 6L deterministic scores differ; "
            "the parent dashboard expects the deterministic baseline to be shared."
        )

    return scores["6-Letter Wordle"]


def fmt_wordle(value):
    return "—" if pd.isna(value) else f"{float(value):.4f}"


def fmt_external(value):
    if pd.isna(value):
        return "—"

    value = float(value)
    if 0 <= value <= 1:
        return f"{value * 100:.1f}%"
    return f"{value:.1f}"


def fmt_rank(value):
    return "—" if pd.isna(value) else str(int(value))


def build_combined_rows(wordle, deterministic_score):
    rows = []

    # Deterministic solver is a reference baseline, not a ranked model.
    rows.append(
        f"""
<tr class="bg-blue-50">
  <td class="px-5 py-4">—</td>
  <td class="px-5 py-4 font-bold">Deterministic Solver (Max Possible)<sup>*</sup></td>
  <td class="px-5 py-4 text-right score">1.0000</td>
  <td class="px-5 py-4 text-right">Nil</td>
  <td class="px-5 py-4 text-right">1.0000</td>
  <td class="px-5 py-4 text-right">1.0000</td>
</tr>
"""
    )

    for _, r in wordle.iterrows():
        rows.append(
            f"""
<tr class="{'rank-1' if pd.notna(r['rank']) and int(r['rank']) == 1 else ''}">
  <td class="px-5 py-4">{fmt_rank(r['rank'])}</td>
  <td class="px-5 py-4 {'font-bold' if pd.notna(r['rank']) and int(r['rank']) == 1 else 'font-semibold'}">
    {html_lib.escape(str(r['model']))}
  </td>
  <td class="px-5 py-4 text-right score">{float(r['overall']):.4f}</td>
  <td class="px-5 py-4 text-right">{float((r['st_5l'] + r['st_6l']) / 2):.4f}</td>
  <td class="px-5 py-4 text-right">{float((r['mt_5l'] + r['mt_6l']) / 2):.4f}</td>
  <td class="px-5 py-4 text-right">{float((r['cf_5l'] + r['cf_6l']) / 2):.4f}</td>
</tr>
"""
        )

    return "\n".join(rows)


def build_external_rows(wordle, external, deterministic_score):
    df = wordle[["model", "overall", "rank"]].merge(
        external,
        on="model",
        how="left",
    )

    rows = []

    # Deterministic solver is a reference baseline. It is not included in
    # model ranking, so its rank is shown as an em dash.
    cells = [
        '<td class="py-3 px-4 border-b font-bold">Deterministic Solver (Max Possible)<sup>*</sup></td>',
        f'<td class="py-3 px-3 border-b text-center bg-blue-50 font-bold">100.0%</td>',
        '<td class="py-3 px-3 border-b text-center bg-blue-50 font-bold">—</td>',
    ]

    for benchmark in [
        "Artificial Analysis Index",
        "GPQA Diamond",
        "MMLU-Pro",
        "HLE",
    ]:
        cells.append(
            '<td class="py-3 px-3 border-b text-center">100.0%</td>'
        )
        cells.append(
            '<td class="py-3 px-3 border-b text-center">—</td>'
        )

    rows.append(
        '<tr class="bg-blue-50">'
        + "".join(cells)
        + '</tr>'
    )

    for _, r in df.iterrows():
        cells = [
            f'<td class="py-3 px-4 border-b font-bold">{html_lib.escape(str(r["model"]))}</td>',
            f'<td class="py-3 px-3 border-b text-center bg-blue-50 font-bold">{float(r["overall"])* 100:.1f}</td>',
            f'<td class="py-3 px-3 border-b text-center bg-blue-50 font-bold">{fmt_rank(r["rank"])}</td>',
        ]

        for benchmark in [
            "Artificial Analysis Index",
            "GPQA Diamond",
            "MMLU-Pro",
            "HLE",
        ]:
            score = fmt_external(r[benchmark])
            rank = fmt_rank(r[f"{benchmark}_rank"])

            cells.append(
                f'<td class="py-3 px-3 border-b text-center">{score}</td>'
            )
            cells.append(
                f'<td class="py-3 px-3 border-b text-center">{rank}</td>'
            )

        rows.append(
            '<tr class="hover:bg-gray-50">'
            + "".join(cells)
            + '</tr>'
        )

    return "\n".join(rows)


def _load_resource_csv(path, benchmark_label):
    """Load one benchmark's resource_metrics.csv using the same field conventions as 5L/6L generators."""
    if not path.exists():
        print(f"Warning: Could not find {path} ({benchmark_label} resource metrics).")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: Could not read {path}: {exc}")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "task" in df.columns and "task_id" not in df.columns:
        df = df.rename(columns={"task": "task_id"})
    if "model" in df.columns and "model_name" not in df.columns:
        df = df.rename(columns={"model": "model_name"})

    required = {"task_id", "model_name", "score", "cost_usd", "time_seconds", "output_tokens"}
    missing = required - set(df.columns)
    if missing:
        print(f"Warning: {path} is missing resource columns: {sorted(missing)}")
        return pd.DataFrame()

    df = df.copy()
    df["model_id"] = df["model_name"].map(canonical_model_id)
    for col in ["score", "cost_usd", "time_seconds", "output_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["model_id", "task_id", "score", "cost_usd", "time_seconds", "output_tokens"])
    df["benchmark"] = benchmark_label
    return df


def _aggregate_resource_benchmark(df, required_tasks):
    """Aggregate resource metrics using the individual dashboard convention."""
    if df.empty:
        return pd.DataFrame()

    df = df[df["task_id"].isin(required_tasks)].copy()
    task_counts = df.groupby("model_id")["task_id"].nunique()
    complete_models = task_counts[task_counts == len(required_tasks)].index
    df = df[df["model_id"].isin(complete_models)].copy()

    if df.empty:
        return pd.DataFrame()

    # Match generate-5l.py: average duplicate runs within each model/task,
    # then equally average the three task scores and sum resource usage.
    task_level = df.groupby(["model_id", "task_id"], as_index=False).agg({
        "score": "mean",
        "cost_usd": "sum",
        "time_seconds": "sum",
        "output_tokens": "sum",
    })

    agg = task_level.groupby("model_id").agg({
        "score": "mean",
        "cost_usd": "sum",
        "time_seconds": "sum",
        "output_tokens": "sum",
    }).reset_index()

    return agg.rename(columns={
        "cost_usd": "cost",
        "time_seconds": "time_seconds",
        "output_tokens": "tokens",
    })


def load_combined_resource_metrics(common_model_ids=None):
    """Load and aggregate the single combined resource_metrics CSV.

    A model is eligible when the combined file contains all three resource
    task IDs: single-turn, multi-turn, and cognitive-flexibility. The number
    of rows is not used as an eligibility requirement; duplicate rows/runs
    within a task are averaged for score and summed for resource usage,
    matching the individual resource-metrics convention.
    """
    required_tasks = {
        "single-turn",
        "multi-turn",
        "cognitive-flexibility",
    }

    if not RESOURCE_COMBINED.exists():
        print(f"Warning: Could not find combined resource metrics: {RESOURCE_COMBINED}")
        return []

    df = _load_resource_csv(RESOURCE_COMBINED, "combined")
    if df.empty:
        return []

    # Normalize task names defensively in case the combined file contains
    # the original 6L spelling.
    df["task_id"] = (
        df["task_id"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "single-turn-v2": "single-turn",
            "single-turn-larger": "single-turn",
        })
    )

    df = df[df["task_id"].isin(required_tasks)].copy()

    # Eligibility is based only on presence of every required task.
    # Do NOT require exactly 3 or 6 rows: duplicate runs are valid.
    task_counts = df.groupby("model_id")["task_id"].nunique()
    eligible_ids = set(
        task_counts[task_counts == len(required_tasks)].index
    )

    print("Complete combined-resource models:", sorted(eligible_ids))

    if not eligible_ids:
        print(
            "Warning: No model has all three required resource tasks "
            "in resource_metrics_combined.csv."
        )
        return []

    df = df[df["model_id"].isin(eligible_ids)].copy()

    # For each model/task, average duplicate score observations and sum the
    # actual resource footprint of those observations.
    task_level = df.groupby(
        ["model_id", "task_id"], as_index=False
    ).agg({
        "score": "mean",
        "cost_usd": "sum",
        "time_seconds": "sum",
        "output_tokens": "sum",
    })

    # Aggregate the three resource tasks for each model.
    agg = task_level.groupby("model_id", as_index=False).agg({
        "score": "mean",
        "cost_usd": "sum",
        "time_seconds": "sum",
        "output_tokens": "sum",
    })

    rows = []
    for _, row in agg.iterrows():
        score = float(row["score"])
        cost = float(row["cost_usd"])
        time_mins = float(row["time_seconds"]) / 60.0
        tokens = float(row["output_tokens"])

        rows.append({
            "model_id": row["model_id"],
            "model_name": display_model_name(row["model_id"]),
            "score": score,
            "cost": cost,
            "time_mins": time_mins,
            "tokens": tokens,
            "score_per_dollar": score / cost if cost > 0 else 0.0,
            "score_per_min": score / time_mins if time_mins > 0 else 0.0,
            "score_per_10k_tokens": (
                (score / tokens) * 10000 if tokens > 0 else 0.0
            ),
        })

    print(
        f"Resource-efficiency model set: {len(rows)} models "
        "with all three resource tasks present."
    )
    return rows


def build_operational_highlights_html(data):
    if not data:
        return '<div class="mt-6 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">No model has complete resource coverage in both 5L and 6L resource files.</div>'

    top_dollar = max(data, key=lambda x: x["score_per_dollar"])
    top_speed = max(data, key=lambda x: x["score_per_min"])
    top_dense = max(data, key=lambda x: x["score_per_10k_tokens"])

    return f"""
    <div class="mt-6 grid md:grid-cols-3 gap-6">
      <div class="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-2xl p-6 shadow-sm">
        <div class="text-green-800 text-xs font-bold uppercase tracking-wider mb-1">Most Economical Genius</div>
        <div class="text-gray-600 text-sm mb-4">Highest Score per Dollar</div>
        <div class="text-2xl font-black text-gray-900 mb-1">{html_lib.escape(top_dollar['model_name'])}</div>
        <div class="text-green-700 font-semibold">{top_dollar['score_per_dollar']:.3f} <span class="text-sm font-normal text-green-600">points/$</span></div>
      </div>
      <div class="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-2xl p-6 shadow-sm">
        <div class="text-blue-800 text-xs font-bold uppercase tracking-wider mb-1">Fastest Decision Maker</div>
        <div class="text-gray-600 text-sm mb-4">Highest Score per Minute</div>
        <div class="text-2xl font-black text-gray-900 mb-1">{html_lib.escape(top_speed['model_name'])}</div>
        <div class="text-blue-700 font-semibold">{top_speed['score_per_min']:.3f} <span class="text-sm font-normal text-blue-600">points/min</span></div>
      </div>
      <div class="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-2xl p-6 shadow-sm">
        <div class="text-purple-800 text-xs font-bold uppercase tracking-wider mb-1">Most Concise Thinker</div>
        <div class="text-gray-600 text-sm mb-4">Highest Score per 10k Tokens</div>
        <div class="text-2xl font-black text-gray-900 mb-1">{html_lib.escape(top_dense['model_name'])}</div>
        <div class="text-purple-700 font-semibold">{top_dense['score_per_10k_tokens']:.3f} <span class="text-sm font-normal text-purple-600">points/10k</span></div>
      </div>
    </div>
    """


def build_resource_efficiency_html(data):
    if not data:
        return '<div class="mt-6 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">No model has complete resource coverage in both 5L and 6L resource files.</div>'

    max_bfb = max(m["score_per_dollar"] for m in data) or 1.0
    max_vel = max(m["score_per_min"] for m in data) or 1.0
    max_den = max(m["score_per_10k_tokens"] for m in data) or 1.0
    for m in data:
        m["combined_efficiency"] = (
            (m["score_per_dollar"] / max_bfb) * 0.33
            + (m["score_per_min"] / max_vel) * 0.34
            + (m["score_per_10k_tokens"] / max_den) * 0.33
        ) * 100
    data = sorted(data, key=lambda x: x["combined_efficiency"], reverse=True)

    rows = []
    for rank, row in enumerate(data, 1):
        rank_style = "text-yellow-600 font-bold text-lg" if rank == 1 else ("text-gray-500 font-bold text-lg" if rank == 2 else ("text-yellow-800 font-bold text-lg" if rank == 3 else "text-gray-700 font-semibold"))
        rows.append(f"""
        <tr class="hover:bg-gray-50 transition-colors">
          <td class="py-3 px-4 border-b {rank_style}">{rank}</td>
          <td class="py-3 px-4 border-b font-bold text-gray-800">{html_lib.escape(row['model_name'])}</td>
          <td class="py-3 px-4 border-b font-black text-indigo-600">{row['combined_efficiency']:.1f} <span class="text-xs font-normal text-gray-400">/ 100</span></td>
          <td class="py-3 px-4 border-b font-mono text-gray-600">${row['cost']:.4f}</td>
          <td class="py-3 px-4 border-b font-bold text-green-600">{row['score_per_dollar']:.3f}</td>
          <td class="py-3 px-4 border-b font-mono text-gray-600">{row['time_mins']:.1f}m</td>
          <td class="py-3 px-4 border-b font-bold text-blue-600">{row['score_per_min']:.4f}</td>
          <td class="py-3 px-4 border-b font-bold text-purple-600">{row['score_per_10k_tokens']:.3f}</td>
        </tr>
        """)

    return f"""
    <div class="mt-6 bg-white border border-slate-200 rounded-2xl shadow-sm table-wrap">
      <table class="w-full text-left border-collapse min-w-[900px]">
        <thead>
          <tr class="bg-slate-50 text-xs text-slate-500 font-semibold uppercase tracking-wider">
            <th class="py-3 px-4 border-b">Rank</th>
            <th class="py-3 px-4 border-b">Model</th>
            <th class="py-3 px-4 border-b text-indigo-700">Combined Score</th>
            <th class="py-3 px-4 border-b">Cost ($)</th>
            <th class="py-3 px-4 border-b">Bang-for-Buck</th>
            <th class="py-3 px-4 border-b">Time (Mins)</th>
            <th class="py-3 px-4 border-b">Velocity</th>
            <th class="py-3 px-4 border-b">Info Density</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <p class="mt-3 text-xs text-slate-500">The efficiency score uses 33% cost efficiency, 34% speed, and 33% information density, with each raw efficiency metric normalized to the maximum among eligible combined models.</p>
    """


def main():
    for path in [FILE_5L, FILE_6L, FILE_EXTERNAL, FILE_SOLVER]:
        require_file(path)

    print(f"5L source: {FILE_5L}")
    print(f"6L source: {FILE_6L}")
    print(f"External benchmark source: {FILE_EXTERNAL}")
    print(f"Deterministic solver source: {FILE_SOLVER}")
    print(f"Combined resource source: {RESOURCE_COMBINED}")

    wordle = load_wordle_scores()

    # The front-page model set is data-driven. External benchmark context is
    # optional and is shown only for these dynamically selected models.
    global DISPLAY_MODELS_DYNAMIC
    DISPLAY_MODELS_DYNAMIC = wordle["model"].tolist()

    external = load_external_benchmarks()
    deterministic_score = load_deterministic_solver_score()
    print(f"Deterministic solver score: {deterministic_score}")

    resource_metrics = load_combined_resource_metrics()
    operational_highlights = build_operational_highlights_html(resource_metrics)
    resource_efficiency = build_resource_efficiency_html(resource_metrics)

    output = TEMPLATE_HTML
    output = output.replace(
        "{{COMBINED_ROWS}}",
        build_combined_rows(wordle, deterministic_score),
    )
    output = output.replace(
        "{{EXTERNAL_ROWS}}",
        build_external_rows(wordle, external, deterministic_score),
    )
    output = output.replace("{{MODEL_COUNT}}", str(len(wordle)))
    output = output.replace("{{OPERATIONAL_HIGHLIGHTS}}", operational_highlights)
    output = output.replace("{{RESOURCE_EFFICIENCY}}", resource_efficiency)

    OUTPUT_FILE.write_text(
        output,
        encoding="utf-8",
    )

    print("\nWordle Combined ranking:")
    print(
        wordle[
            ["rank", "model", "score_5l", "score_6l", "overall"]
        ].to_string(index=False)
    )

    print(f"\nGenerated: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
