import json
import os
import glob

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TASKS = [
    {
        "id": "multi-turn",
        "title": "Multi-turn Wordle",
        "description": "Evaluating executive function and information gain across multi-turn gameplay.",
    },
    {
        "id": "single-turn",
        "title": "Single-turn Wordle",
        "description": "Evaluating executive function from a single Wordle decision.",
    },
    {
        "id": "cognitive-flexibility",
        "title": "Cognitive Flexibility",
        "description": "Evaluating adaptive reasoning and cognitive flexibility.",
    },
]

DATA_ROOT = "data"
TEMPLATE_DIR = "templates"

HOME_TEMPLATE = os.path.join(TEMPLATE_DIR, "home.html")
LEADERBOARD_TEMPLATE = os.path.join(TEMPLATE_DIR, "leaderboard.html")

HOME_OUTPUT = "index.html"

ROW_PLACEHOLDER = "{{TABLE_ROWS}}"
TITLE_PLACEHOLDER = "{{TITLE}}"
DESCRIPTION_PLACEHOLDER = "{{DESCRIPTION}}"
NAV_PLACEHOLDER = "{{NAVIGATION}}"
CARDS_PLACEHOLDER = "{{TASK_CARDS}}"
COGNITIVE_DEGRADATION_PLACEHOLDER = "{{COGNITIVE_DEGRADATION}}"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_json_results(folder):
    results = []
    json_files = sorted(glob.glob(os.path.join(folder, "*.json")))

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # --- THE FIX: Normalize the score key across all tasks ---
            # If the JSON uses 'overall_score', map it to 'overall_benchmark_score'
            if "overall_score" in data and "overall_benchmark_score" not in data:
                data["overall_benchmark_score"] = data["overall_score"]
                
            data["model_name"] = os.path.splitext(os.path.basename(file))[0]
            results.append(data)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    # Now the sort will work perfectly for both naming conventions!
    results.sort(
        key=lambda x: x.get("overall_benchmark_score", 0),
        reverse=True,
    )
    return results

def build_table(results):
    html = ""
    for rank, row in enumerate(results, start=1):
        score = row.get("overall_benchmark_score", 0)
        win_rate = row.get("win_rate", 0) * 100
        violations = row.get("avg_violations_per_game", 0)
        error_rate = row.get("api_error_rate", 0) * 100
        penalty = row.get("penalty_applied", False)

        if rank == 1:
            rank_style = "text-yellow-600 font-bold text-lg"
        elif rank == 2:
            rank_style = "text-gray-500 font-bold text-lg"
        elif rank == 3:
            rank_style = "text-yellow-800 font-bold text-lg"
        else:
            rank_style = "text-gray-700"

        violation_style = (
            "text-green-600"
            if violations == 0
            else "text-red-600 font-semibold"
        )

        penalty_html = (
            '<span class="bg-red-100 text-red-800 text-xs font-medium px-2 py-1 rounded">Yes</span>'
            if penalty
            else "-"
        )

        html += f"""
<tr class="hover:bg-gray-50">
<td class="py-4 px-6 border-b {rank_style}">{rank}</td>
<td class="py-4 px-6 border-b font-semibold">{row['model_name']}</td>
<td class="py-4 px-6 border-b font-mono">{score:.4f}</td>
<td class="py-4 px-6 border-b">{win_rate:.1f}%</td>
<td class="py-4 px-6 border-b {violation_style}">{violations:.3f}</td>
<td class="py-4 px-6 border-b">{error_rate:.1f}%</td>
<td class="py-4 px-6 border-b text-center">{penalty_html}</td>
</tr>
"""
    return html


def navigation(active):
    links = ['<a href="index.html" class="hover:text-blue-600">Home</a>']
    for task in TASKS:
        if task["id"] == active:
            links.append(
                f'<span class="font-bold text-blue-600">{task["title"]}</span>'
            )
        else:
            links.append(
                f'<a href="{task["id"]}.html" class="hover:text-blue-600">{task["title"]}</a>'
            )
    return " | ".join(links)


# --------------------------------------------------
# Insights Helpers
# --------------------------------------------------

def calculate_cognitive_degradation():
    """Calculates the difference in rule violations between multi-turn and single-turn."""
    single_turn_data = load_json_results(os.path.join(DATA_ROOT, "single-turn"))
    multi_turn_data = load_json_results(os.path.join(DATA_ROOT, "multi-turn"))

    st_dict = {m["model_name"]: m.get("avg_violations_per_game", 0) for m in single_turn_data}
    mt_dict = {m["model_name"]: m.get("avg_violations_per_game", 0) for m in multi_turn_data}

    degradation_results = []
    
    for model_name in st_dict.keys():
        if model_name in mt_dict:
            st_violations = st_dict[model_name]
            mt_violations = mt_dict[model_name]
            
            degradation = mt_violations - st_violations
            
            degradation_results.append({
                "model_name": model_name,
                "st_violations": st_violations,
                "mt_violations": mt_violations,
                "degradation": degradation
            })

    # Sort so models with the LEAST degradation (most stable) are at the top
    degradation_results.sort(key=lambda x: x["degradation"])
    return degradation_results


def build_degradation_html(results):
    if not results:
        return "<p class='text-gray-500'>Not enough data to calculate Cognitive Degradation across tasks.</p>"

    html = """
    <div class="bg-white rounded-lg shadow p-6 border mb-8">
        <h3 class="text-xl font-bold mb-2">Cognitive Degradation Index (Working Memory Stress Test)</h3>
        <p class="text-gray-600 mb-6 text-sm">Measures how much a model's rule adherence degrades when transitioning from a single decision to multi-turn gameplay. A lower score indicates better context stability.</p>
        
        <table class="w-full text-left">
            <thead>
                <tr class="bg-gray-50 uppercase text-xs text-gray-500">
                    <th class="py-3 px-4 border-b">Rank</th>
                    <th class="py-3 px-4 border-b">Model</th>
                    <th class="py-3 px-4 border-b">Single-Turn Violations</th>
                    <th class="py-3 px-4 border-b">Multi-Turn Violations</th>
                    <th class="py-3 px-4 border-b">Degradation Score</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for rank, row in enumerate(results, start=1):
        deg = row['degradation']
        if deg <= 0:
            deg_color = "text-green-600 font-bold"
            deg_text = f"{deg:.3f} (Stable)"
        elif deg < 0.1:
            deg_color = "text-yellow-600 font-bold"
            deg_text = f"+{deg:.3f} (Slight Drift)"
        else:
            deg_color = "text-red-600 font-bold"
            deg_text = f"+{deg:.3f} (Severe Drift)"

        html += f"""
            <tr class="hover:bg-gray-50">
                <td class="py-3 px-4 border-b font-medium text-gray-700">{rank}</td>
                <td class="py-3 px-4 border-b font-semibold">{row['model_name']}</td>
                <td class="py-3 px-4 border-b">{row['st_violations']:.3f}</td>
                <td class="py-3 px-4 border-b">{row['mt_violations']:.3f}</td>
                <td class="py-3 px-4 border-b {deg_color}">{deg_text}</td>
            </tr>
        """
    html += "</tbody></table></div>"
    return html



# ... (Your existing code down to Insights Helpers) ...

# --------------------------------------------------
# Insights Helpers
# --------------------------------------------------

# ... (Keep your existing calculate_cognitive_degradation and build_degradation_html functions) ...

def calculate_strategy_quadrant():
    """Maps models into quadrants based on Win Rate and Info Gain Score."""
    # We use multi-turn as it's the best indicator of sustained strategy
    mt_data = load_json_results(os.path.join(DATA_ROOT, "multi-turn"))
    
    if not mt_data:
        return []

    # Calculate medians to draw the quadrant lines dynamically based on the cohort
    scores = [m.get("overall_benchmark_score", 0) for m in mt_data]
    win_rates = [m.get("win_rate", 0) for m in mt_data]
    
    median_score = sorted(scores)[len(scores)//2] if scores else 0
    median_win_rate = sorted(win_rates)[len(win_rates)//2] if win_rates else 0

    quadrants = {
        "masters": [],      # High Win, High Score
        "lucky": [],        # High Win, Low Score
        "overthinkers": [], # Low Win, High Score
        "struggling": []    # Low Win, Low Score
    }

    for model in mt_data:
        score = model.get("overall_benchmark_score", 0)
        win_rate = model.get("win_rate", 0)
        name = model["model_name"]

        if win_rate >= median_win_rate and score >= median_score:
            quadrants["masters"].append(name)
        elif win_rate >= median_win_rate and score < median_score:
            quadrants["lucky"].append(name)
        elif win_rate < median_win_rate and score >= median_score:
            quadrants["overthinkers"].append(name)
        else:
            quadrants["struggling"].append(name)
            
    return quadrants

def build_strategy_html(quadrants):
    if not any(quadrants.values()):
        return ""

    html = """
    <div class="bg-white rounded-lg shadow p-6 border mb-8">
        <h3 class="text-xl font-bold mb-2">Strategy vs. Brute-Force Matrix (Multi-Turn)</h3>
        <p class="text-gray-600 mb-6 text-sm">Compares a model's ability to win against its Information Gain strategy. Placed relative to the cohort median.</p>
        
        <div class="flex items-center gap-4">
            
            <div class="flex flex-col justify-between items-center h-[528px] text-xs font-bold text-gray-400 uppercase tracking-wide py-4 select-none" style="writing-mode: vertical-rl; transform: rotate(180deg);">
                <span>&larr; Win Rate &rarr;</span>
            </div>

            <div class="grid grid-cols-2 gap-4 flex-1">
                <div class="border-2 border-dashed border-gray-300 bg-gray-50 p-4 rounded-lg h-64 flex flex-col relative">
                    <span class="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">Lucky Guessers</span>
                    <span class="text-xs text-gray-400 mb-2">High Win Rate, Low Strategy</span>
                    <ul class="text-sm font-semibold text-blue-700">
    """
    for model in quadrants["lucky"]: html += f"<li>{model}</li>"
    
    html += """
                    </ul>
                </div>
                
                <div class="border-2 border-blue-200 bg-blue-50 p-4 rounded-lg h-64 flex flex-col relative">
                    <span class="text-xs font-bold uppercase tracking-wider text-blue-600 mb-2">Reasoning Masters</span>
                    <span class="text-xs text-blue-400 mb-2">High Win Rate, High Strategy</span>
                    <ul class="text-sm font-bold text-gray-900">
    """
    for model in quadrants["masters"]: html += f"<li>{model}</li>"

    html += """
                    </ul>
                </div>

                <div class="border-2 border-dashed border-gray-200 bg-gray-50 p-4 rounded-lg h-64 flex flex-col relative opacity-75">
                    <span class="text-xs font-bold uppercase tracking-wider text-gray-400 mb-2">Struggling</span>
                    <span class="text-xs text-gray-400 mb-2">Low Win Rate, Low Strategy</span>
                    <ul class="text-sm font-medium text-gray-600">
    """
    for model in quadrants["struggling"]: html += f"<li>{model}</li>"

    html += """
                    </ul>
                </div>

                <div class="border-2 border-dashed border-gray-300 bg-gray-50 p-4 rounded-lg h-64 flex flex-col relative">
                    <span class="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">Over-thinkers</span>
                    <span class="text-xs text-gray-400 mb-2">Low Win Rate, High Strategy</span>
                    <ul class="text-sm font-medium text-purple-700">
    """
    for model in quadrants["overthinkers"]: html += f"<li>{model}</li>"

    html += """
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="flex justify-between text-xs font-bold text-gray-400 mt-4 pl-10 pr-2 uppercase tracking-wide">
            <span>&larr; Lower Info Gain</span>
            <span>Strategy Score (Overall Benchmark)</span>
            <span>Higher Info Gain &rarr;</span>
        </div>
    </div>
    """
    return html



# ... (Your existing code down to calculate_strategy_quadrant / build_strategy_html) ...

def calculate_compliance_score():
    """Calculates a Reliability Score and assigns a Deployment Tier."""
    mt_data = load_json_results(os.path.join(DATA_ROOT, "multi-turn"))
    
    compliance_results = []
    
    for model in mt_data:
        error_rate = model.get("api_error_rate", 0)
        penalty = model.get("penalty_applied", False)
        
        # Calculate Base Reliability Score (0-100)
        reliability_score = max(0, 100 - (error_rate * 100))
        
        # Determine Tier
        if penalty:
            tier = "Penalty"
            color = "bg-red-50 border-red-200 text-red-800"
            icon = "☠️"
        elif reliability_score >= 99.0:
            tier = "Production Ready"
            color = "bg-green-50 border-green-200 text-green-800"
            icon = "🛡️"
        elif reliability_score >= 90.0:
            tier = "Stable"
            color = "bg-blue-50 border-blue-200 text-blue-800"
            icon = "✅"
        else:
            tier = "Fragile"
            color = "bg-yellow-50 border-yellow-200 text-yellow-800"
            icon = "⚠️"

        compliance_results.append({
            "model_name": model["model_name"],
            "reliability_score": reliability_score,
            "error_rate": error_rate,
            "penalty": penalty,
            "tier": tier,
            "color": color,
            "icon": icon
        })

    # Sort so the highest reliability is at the top
    compliance_results.sort(key=lambda x: x["reliability_score"], reverse=True)
    return compliance_results

def build_compliance_html(results):
    if not results:
        return ""

    html = """
    <div class="bg-white rounded-lg shadow p-6 border mb-12">
        <h3 class="text-xl font-bold mb-2">Deployment Safety & Compliance</h3>
        <p class="text-gray-600 mb-6 text-sm">Ranks models based on format adherence and API stability. Models that trigger the >20% error penalty are placed in the Penalty category.</p>
        
        <div class="grid md:grid-cols-2 gap-4">
    """
    
    for row in results:
        html += f"""
            <div class="border rounded-lg p-4 flex items-center justify-between {row['color']}">
                <div class="flex items-center gap-3">
                    <span class="text-2xl">{row['icon']}</span>
                    <div>
                        <h4 class="font-bold text-lg leading-tight">{row['model_name']}</h4>
                        <span class="text-xs font-semibold uppercase tracking-wider opacity-75">{row['tier']}</span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-2xl font-black">{row['reliability_score']:.1f}<span class="text-lg text-gray-500 font-medium">%</span></div>
                    <div class="text-xs opacity-75 mt-1">API Errors: {row['error_rate']*100:.1f}%</div>
                </div>
            </div>
        """
        
    html += """
        </div>
    </div>
    """
    return html









# ... (Your existing code down to calculate_compliance_score / build_compliance_html) ...

# --------------------------------------------------
# 4. Executive Profile (Radar Charts)
# --------------------------------------------------

def calculate_executive_profiles(top_n=5):
    """Normalizes scores and returns the Top N models for radar charts."""
    tasks = ["single-turn", "multi-turn", "cognitive-flexibility"]
    model_scores = {}
    task_maxes = {t: 0.0001 for t in tasks} # Prevent division by zero
    
    # 1. Gather raw scores and find maximums for normalization
    for task in tasks:
        data = load_json_results(os.path.join(DATA_ROOT, task))
        for model in data:
            name = model["model_name"]
            score = model.get("overall_benchmark_score", 0)
            
            if name not in model_scores:
                model_scores[name] = {"single-turn": 0, "multi-turn": 0, "cognitive-flexibility": 0}
            
            model_scores[name][task] = score
            if score > task_maxes[task]:
                task_maxes[task] = score

    # 2. Normalize to a 0-100 scale based on the top performer of each task
    normalized_profiles = {}
    for name, scores in model_scores.items():
        normalized_profiles[name] = {
            "single-turn": round((scores["single-turn"] / task_maxes["single-turn"]) * 100, 1),
            "multi-turn": round((scores["multi-turn"] / task_maxes["multi-turn"]) * 100, 1),
            "cognitive-flexibility": round((scores["cognitive-flexibility"] / task_maxes["cognitive-flexibility"]) * 100, 1)
        }
        
    # 3. Sort by average normalized score and slice the Top N
    def get_average(item):
        scores = item[1]
        return sum(scores.values()) / len(scores)

    # Sort descending based on the average score
    sorted_profiles = sorted(normalized_profiles.items(), key=get_average, reverse=True)
    
    # Return only the top N as a dictionary
    return dict(sorted_profiles[:top_n])

def build_executive_profile_html(profiles):
    if not profiles:
        return ""

    # Include Chart.js from CDN and setup the grid
    html = """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <div class="bg-white rounded-lg shadow p-6 border mb-12">
        <h3 class="text-xl font-bold mb-2">Executive Cognitive Profiles</h3>
        <p class="text-gray-600 mb-6 text-sm">Visualizes normalized performance across all three benchmark dimensions. 100 represents the state-of-the-art maximum for that specific task.</p>
        
        <div class="grid md:grid-cols-3 gap-6">
    """
    
    # Generate a canvas for each model
    chart_scripts = "<script>\n"
    
    for i, (model_name, scores) in enumerate(profiles.items()):
        canvas_id = f"radarChart_{i}"
        
        html += f"""
            <div class="border rounded-lg p-4 flex flex-col items-center bg-gray-50">
                <h4 class="font-bold text-gray-800 mb-2">{model_name}</h4>
                <div class="w-full relative" style="max-width: 250px;">
                    <canvas id="{canvas_id}"></canvas>
                </div>
            </div>
        """
        
        # Build the JavaScript to render this specific chart
        # Using a distinct color to make it look professional (a nice Kaggle blue)
        chart_scripts += f"""
        new Chart(document.getElementById('{canvas_id}'), {{
            type: 'radar',
            data: {{
                labels: ['Single-Turn', 'Multi-Turn', 'Cognitive Flex'],
                datasets: [{{
                    label: 'Normalized Score',
                    data: [{scores['single-turn']}, {scores['multi-turn']}, {scores['cognitive-flexibility']}],
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                scales: {{
                    r: {{
                        angleLines: {{ display: true }},
                        suggestedMin: 0,
                        suggestedMax: 100,
                        ticks: {{ stepSize: 20, display: false }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});
        """
        
    html += """
        </div>
    </div>
    """
    chart_scripts += "</script>\n"
    
    # Combine HTML and the executing scripts
    return html + chart_scripts


def calculate_cognitive_pillars():
    """Maps JSON data to the 3 pillars of Executive Function and normalizes for charting."""
    st_data = load_json_results(os.path.join(DATA_ROOT, "single-turn"))
    mt_data = load_json_results(os.path.join(DATA_ROOT, "multi-turn"))
    cf_data = load_json_results(os.path.join(DATA_ROOT, "cognitive-flexibility"))

    # Create lookups
    st_dict = {m["model_name"]: m.get("avg_violations_per_game", 0) for m in st_data}
    mt_dict = {m["model_name"]: m.get("avg_violations_per_game", 0) for m in mt_data}
    cf_viol_dict = {m["model_name"]: m.get("avg_violations_per_game", 0) for m in cf_data}
    cf_dict = {m["model_name"]: m.get("overall_benchmark_score", 0) for m in cf_data}

    # Unique list of all models
    models = set(list(st_dict.keys()) + list(mt_dict.keys()) + list(cf_dict.keys()))

    results = []
    max_deg = 0.0001
    max_viol = 0.0001
    max_cf = 0.0001

    # Step 1: Extract Raw Data
    for model in models:
        st_v = st_dict.get(model, 0)
        mt_v = mt_dict.get(model, 0)
        cf_v = cf_viol_dict.get(model, 0)
        cf_score = cf_dict.get(model, 0)

        # Working Memory proxy (Multi-turn violations minus Single-turn)
        degradation = max(0, mt_v - st_v) 
        
        # Inhibitory Control proxy (Spike in violations during CF task vs MT task)
        inhibition = max(0, cf_v - mt_v)

        # Track maximums across the cohort for scaling
        max_deg = max(max_deg, degradation)
        max_viol = max(max_viol, inhibition)
        max_cf = max(max_cf, cf_score)

        results.append({
            "model_name": model,
            "raw_wm_deg": degradation,
            "raw_cf_score": cf_score,
            "raw_inhibition": inhibition
        })

    # Step 2: Normalize to 0-100 (where 100 is always the BEST performance)
    for r in results:
        # WM & Inhibition: Lower is better. So 0 violations = 100%. 
        r["norm_wm"] = max(0, 100 - ((r["raw_wm_deg"] / max_deg) * 100)) if max_deg > 0 else 100
        r["norm_inhibition"] = max(0, 100 - ((r["raw_inhibition"] / max_viol) * 100)) if max_viol > 0 else 100
        
        # CF: Higher is better.
        r["norm_cf"] = (r["raw_cf_score"] / max_cf) * 100 if max_cf > 0 else 0

        # Create an average score strictly for ranking the Top 5
        r["avg_norm"] = (r["norm_wm"] + r["norm_cf"] + r["norm_inhibition"]) / 3

    # Sort by the highest average normalized score
    results.sort(key=lambda x: x["avg_norm"], reverse=True)
    return results


def build_pillars_html(results):
    if not results:
        return ""

    # Build the Table (Raw Data)
    html = """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <div class="bg-white rounded-lg shadow p-6 border mb-12">
        <h3 class="text-2xl font-bold mb-2">The 3 Pillars of Executive Function</h3>
        <p class="text-gray-600 mb-6">Mapping benchmark outcomes directly to cognitive science pillars.</p>
	<p class="text-xs text-gray-500 mb-6">
    <span class="font-bold">Working Memory:</span> MT Violations &minus; ST Violations &nbsp;|&nbsp; 
    <span class="font-bold">Cognitive Flex:</span> Task 3 Score &nbsp;|&nbsp; 
    <span class="font-bold">Inhibitory Control:</span> Task 3 Violations &minus; MT Violations
	</p>

        <div class="overflow-x-auto mb-10">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="bg-gray-100 uppercase text-xs text-gray-600 font-semibold tracking-wider">
                        <tr class="bg-gray-100 uppercase text-xs text-gray-600 font-semibold tracking-wider">
                        <th class="py-3 px-4 border-b">Rank</th>
                        <th class="py-3 px-4 border-b">Model</th>
                        <th class="py-3 px-4 border-b">Working Memory (Drift)</th>
                        <th class="py-3 px-4 border-b">Cognitive Flexibility</th>
                        <th class="py-3 px-4 border-b">Inhibitory Control</th>
                        <th class="py-3 px-4 border-b bg-blue-50 text-blue-900 rounded-tr-lg">Overall EF Score (0-100)</th>
                    </tr>
                </thead>
                <tbody>
    """
    for i, r in enumerate(results, start=1):
        html += f"""
            <tr class="hover:bg-gray-50 transition-colors">
                <td class="py-3 px-4 border-b text-gray-700 font-medium">{i}</td>
                <td class="py-3 px-4 border-b font-bold text-gray-900">{r['model_name']}</td>
                <td class="py-3 px-4 border-b text-blue-700">{r['raw_wm_deg']:.3f}</td>
                <td class="py-3 px-4 border-b text-purple-700">{r['raw_cf_score']:.4f}</td>
                <td class="py-3 px-4 border-b text-red-700">{r['raw_inhibition']:.3f}</td>
                <td class="py-3 px-4 border-b bg-blue-50 font-black text-blue-800 text-lg">{r['avg_norm']:.1f}</td>
            </tr>
        """
    html += "</tbody></table></div>"

    # Build the Radar Chart (Top 5 Only)
    top_5 = results[:5]
    
    html += """
        <div class="bg-gray-50 p-6 rounded-lg border">
	#
            <h4 class="font-bold text-xl mb-2 text-center text-gray-800">Top 5 Models: Cognitive Profile Map</h4>
            <p class="text-center text-sm text-gray-500 mb-6">Scores normalized 0-100 (where 100 represents cohort state-of-the-art).</p>
            <div class="w-full max-w-3xl mx-auto relative h-[400px]">
                <canvas id="pillarsChart"></canvas>
            </div>
        </div>
    </div>
    <script>
    """
    
    # Distinct colors for the 5 lines (Blue, Green, Orange, Purple, Red)
    colors = [
        ("rgba(59, 130, 246, 0.1)", "rgba(59, 130, 246, 1)"),   
        ("rgba(16, 185, 129, 0.1)", "rgba(16, 185, 129, 1)"),   
        ("rgba(245, 158, 11, 0.1)", "rgba(245, 158, 11, 1)"),   
        ("rgba(139, 92, 246, 0.1)", "rgba(139, 92, 246, 1)"),   
        ("rgba(239, 68, 68, 0.1)", "rgba(239, 68, 68, 1)")      
    ]

    datasets_js = []
    for i, r in enumerate(top_5):
        bg_color, border_color = colors[i % len(colors)]
        datasets_js.append(f"""
        {{
            label: '{r['model_name']}',
            data: [{r['norm_wm']:.1f}, {r['norm_cf']:.1f}, {r['norm_inhibition']:.1f}],
            backgroundColor: '{bg_color}',
            borderColor: '{border_color}',
            pointBackgroundColor: '{border_color}',
            borderWidth: 2,
            tension: 0.1
        }}
        """)

    html += f"""
    new Chart(document.getElementById('pillarsChart'), {{
        type: 'radar',
        data: {{
            labels: ['Working Memory (Stability)', 'Cognitive Flexibility', 'Inhibitory Control'],
            datasets: [{','.join(datasets_js)}]
        }},
        options: {{
            maintainAspectRatio: false,
            scales: {{
                r: {{
                    angleLines: {{ display: true, color: 'rgba(0,0,0,0.1)' }},
                    grid: {{ color: 'rgba(0,0,0,0.05)' }},
                    suggestedMin: 0,
                    suggestedMax: 100,
                    pointLabels: {{ font: {{ size: 14, weight: 'bold' }}, color: '#374151' }},
                    ticks: {{ display: false }}
                }}
            }},
            plugins: {{
                legend: {{ position: 'bottom', labels: {{ padding: 20, font: {{ size: 12, weight: 'bold' }} }} }},
                tooltip: {{ backgroundColor: 'rgba(0,0,0,0.8)', padding: 12 }}
            }}
        }}
    }});
    </script>
    """
    return html



#leaderboard charts

def get_weighted_scores():

    import pandas as pd

    # 1. Load the CSV you downloaded from Kaggle

    df = pd.read_csv('wordle-bench_leaderboard.csv')
    
    # 2. Filter for your 3 specific tasks
    relevant_tasks = [
        'evaluate_wordle_single_turn_v2', 
        'evaluate_wordle_multi_turn', # Update if your CSV name is exactly 'evaluate_wordle_multi_turn'
        'evaluate_cognitive_flexibility'
    ]
    df_filtered = df[df['Task_Name'].isin(relevant_tasks)]
    
    # 3. Create the pivot table (Models as rows, Tasks as columns)
    grouped = df_filtered.groupby(['Model', 'Task_Name'])['Numerical_Result'].mean().unstack()
    
    # 4. Fill missing data with 0 and apply your weights
    grouped_filled = grouped.fillna(0)
    
    weights = {
        'evaluate_wordle_single_turn_v2': 0.20,
        'evaluate_wordle_multi_turn': 0.30,
        'evaluate_cognitive_flexibility': 0.50
    }
    
    grouped_filled['Combined_Score'] = (
        grouped_filled['evaluate_wordle_single_turn_v2'] * weights['evaluate_wordle_single_turn_v2'] +
        grouped_filled['evaluate_wordle_multi_turn'] * weights['evaluate_wordle_multi_turn'] +
        grouped_filled['evaluate_cognitive_flexibility'] * weights['evaluate_cognitive_flexibility']
    )
    
    return grouped_filled.sort_values(by='Combined_Score', ascending=False)

def build_combined_bar_chart_html(combined_df):
    # We need to pass the raw unweighted data to the frontend 
    # so JavaScript can recalculate the weights dynamically.
    models = combined_df.index.tolist()
    
    # Extract the raw scores for each task per model
    st_scores = combined_df['evaluate_wordle_single_turn_v2'].tolist()
    mt_scores = combined_df['evaluate_wordle_multi_turn'].tolist()
    cf_scores = combined_df['evaluate_cognitive_flexibility'].tolist()

    html = """
    <div class="bg-white rounded-lg shadow p-6 border mb-12">
        <h3 class="text-2xl font-bold mb-2">Aggregate Executive Function Performance</h3>
        <p class="text-gray-600 mb-4 text-sm">Adjust the weights to see how different evaluation priorities affect the leaderboard.</p>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6 p-4 bg-gray-50 rounded-lg border">
            <div>
                <label class="block text-sm font-bold text-gray-700 mb-1">Single-Turn Weight: <span id="stWeightVal" class="text-blue-600">20%</span></label>
                <input type="range" id="stWeight" min="0" max="100" value="20" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
            </div>
            <div>
                <label class="block text-sm font-bold text-gray-700 mb-1">Multi-Turn Weight: <span id="mtWeightVal" class="text-blue-600">30%</span></label>
                <input type="range" id="mtWeight" min="0" max="100" value="30" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
            </div>
            <div>
                <label class="block text-sm font-bold text-gray-700 mb-1">Cognitive Flex Weight: <span id="cfWeightVal" class="text-blue-600">50%</span></label>
                <input type="range" id="cfWeight" min="0" max="100" value="50" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
            </div>
        </div>
        <p id="weightWarning" class="text-red-500 text-sm font-bold hidden mb-4">Weights must sum to 100%!</p>

        <div class="relative w-full" style="height: 400px;">
            <canvas id="combinedChart"></canvas>
        </div>
    </div>
    
    <script>
    window.addEventListener('load', function() {
        // Raw Data from Python
        const rawData = {
            models: """ + json.dumps(models) + """,
            singleTurn: """ + json.dumps(st_scores) + """,
            multiTurn: """ + json.dumps(mt_scores) + """,
            cognitiveFlex: """ + json.dumps(cf_scores) + """
        };

        let myChart = null;

        function updateChart() {
            // Get current slider values
            const wST = parseInt(document.getElementById('stWeight').value);
            const wMT = parseInt(document.getElementById('mtWeight').value);
            const wCF = parseInt(document.getElementById('cfWeight').value);

            // Update text labels
            document.getElementById('stWeightVal').innerText = wST + '%';
            document.getElementById('mtWeightVal').innerText = wMT + '%';
            document.getElementById('cfWeightVal').innerText = wCF + '%';

            const totalWeight = wST + wMT + wCF;
            const warningEl = document.getElementById('weightWarning');

            if (totalWeight !== 100) {
                warningEl.classList.remove('hidden');
                warningEl.innerText = `Weights currently sum to ${totalWeight}%. They must equal 100%.`;
                return; // Stop updating chart if weights are invalid
            } else {
                warningEl.classList.add('hidden');
            }

            // Calculate new combined scores
            let calculatedScores = [];
            for (let i = 0; i < rawData.models.length; i++) {
                const score = (rawData.singleTurn[i] * (wST / 100)) + 
                              (rawData.multiTurn[i] * (wMT / 100)) + 
                              (rawData.cognitiveFlex[i] * (wCF / 100));
                
                calculatedScores.push({
                    model: rawData.models[i],
                    score: score
                });
            }

            // Sort descending by new score
            calculatedScores.sort((a, b) => b.score - a.score);

            const sortedLabels = calculatedScores.map(item => item.model);
            const sortedData = calculatedScores.map(item => item.score);

            // Render or Update Chart
            const ctx = document.getElementById('combinedChart').getContext('2d');
            
            if (myChart) {
                myChart.data.labels = sortedLabels;
                myChart.data.datasets[0].data = sortedData;
                myChart.update();
            } else {
                myChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: sortedLabels,
                        datasets: [{
                            label: 'Weighted Aggregate Score',
                            data: sortedData,
                            backgroundColor: 'rgba(59, 130, 246, 0.8)'
                        }]
                    },
                    options: { 
                        responsive: true, 
                        maintainAspectRatio: false,
                        scales: { y: { beginAtZero: true } }
                    }
                });
            }
        }

        // Attach event listeners to sliders
        document.getElementById('stWeight').addEventListener('input', updateChart);
        document.getElementById('mtWeight').addEventListener('input', updateChart);
        document.getElementById('cfWeight').addEventListener('input', updateChart);

        // Initial render
        updateChart();
    });
    </script>
    """
    return html




import pandas as pd
import math

def load_resource_metrics(csv_path="data/resource_metrics.csv"):
    """Loads resource data from CSV independently of JSON data."""
    resources = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Normalize columns safely so it doesn't matter if they are capitalized
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Fallbacks for header mapping
        if 'task' in df.columns and 'task_id' not in df.columns:
            df = df.rename(columns={'task': 'task_id'})
        if 'model' in df.columns and 'model_name' not in df.columns:
            df = df.rename(columns={'model': 'model_name'})

        for _, row in df.iterrows():
            task = row.get('task_id')
            model = row.get('model_name')
            
            if pd.isna(task) or pd.isna(model): continue
            
            if task not in resources:
                resources[task] = {}
                
            resources[task][model] = {
                'score': float(row.get('score', 0.0) if pd.notna(row.get('score')) else 0.0),
                'input_tokens': float(row.get('input_tokens', 0)),
                'output_tokens': float(row.get('output_tokens', 0)),
                'cost_usd': float(row.get('cost_usd', 0.0)),
                'time_seconds': float(row.get('time_seconds', 0.0))
            }
    return resources

def build_efficiency_html(task_resources):
    """Builds a UI section for Resource Efficiency metrics, strictly from CSV data."""
    valid_results = []
    
    # task_resources is a dict containing only CSV data for this specific task
    for model_name, metrics in task_resources.items():
        try:
            cost = float(metrics.get('cost_usd', 0.0))
            if cost > 0:
                valid_results.append({
                    'model_name': str(model_name),
                    'cost_usd': cost,
                    'time_seconds': float(metrics.get('time_seconds', 0.0)),
                    'output_tokens': float(metrics.get('output_tokens', 0.0)),
                    # STRICTLY extract only 'score'
                    'score': float(metrics.get('score', 0.0))
                })
        except (ValueError, TypeError):
            continue
            
    if not valid_results:
        return ""

    # 1. Calculate raw metrics for all valid models
    metrics_list = []
    for row in valid_results:
        cost = row['cost_usd']
        time_mins = row['time_seconds'] / 60.0
        out_tokens = row['output_tokens']
        score = row['score']
        
        bang_for_buck = score / cost if cost > 0 else 0.0
        velocity = score / time_mins if time_mins > 0 else 0.0
        density = (score / out_tokens) * 10000 if out_tokens > 0 else 0.0
        
        metrics_list.append({
            'model_name': row['model_name'],
            'cost': cost,
            'bang_for_buck': bang_for_buck,
            'time_mins': time_mins,
            'velocity': velocity,
            'density': density
        })

    # 2. Normalize to a 0-1 scale (using the Max value)
    max_bfb = max([m['bang_for_buck'] for m in metrics_list]) if metrics_list else 1.0
    max_vel = max([m['velocity'] for m in metrics_list]) if metrics_list else 1.0
    max_den = max([m['density'] for m in metrics_list]) if metrics_list else 1.0
    
    # Prevent division by zero
    max_bfb = max_bfb if max_bfb > 0 else 1.0
    max_vel = max_vel if max_vel > 0 else 1.0
    max_den = max_den if max_den > 0 else 1.0

    # 3. Define Weights
    WEIGHT_BFB = 0.33   
    WEIGHT_VEL = 0.34   
    WEIGHT_DEN = 0.33  

    # 4. Calculate the Combined Score out of 100
    for m in metrics_list:
        norm_bfb = m['bang_for_buck'] / max_bfb
        norm_vel = m['velocity'] / max_vel
        norm_den = m['density'] / max_den
        
        m['combined_score'] = (norm_bfb * WEIGHT_BFB + norm_vel * WEIGHT_VEL + norm_den * WEIGHT_DEN) * 100

    # 5. Sort the list by Combined Score
    metrics_list.sort(key=lambda x: x['combined_score'], reverse=True)

    # 6. Build the HTML Table
    html = """
    <div class="bg-white rounded-lg shadow p-6 border mt-8 w-full">
        <h3 class="text-xl font-bold mb-2">Resource & Efficiency Metrics</h3>
        <p class="text-gray-600 mb-6 text-sm">Evaluating operational footprint. Sorted by a <strong>Combined Efficiency Score</strong> (40% Cost, 40% Speed, 20% Density).</p>
        
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="bg-gray-50 uppercase text-xs text-gray-500 font-semibold tracking-wider">
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
                <tbody>
    """
    
    for rank, row in enumerate(metrics_list, start=1):
        if rank == 1:
            rank_style = "text-yellow-600 font-bold text-lg"
        elif rank == 2:
            rank_style = "text-gray-500 font-bold text-lg"
        elif rank == 3:
            rank_style = "text-yellow-800 font-bold text-lg"
        else:
            rank_style = "text-gray-700 font-semibold"

        html += f"""
            <tr class="hover:bg-gray-50 transition-colors">
                <td class="py-3 px-4 border-b {rank_style}">{rank}</td>
                <td class="py-3 px-4 border-b font-bold text-gray-800">{row['model_name']}</td>
                <td class="py-3 px-4 border-b font-black text-indigo-600">{row['combined_score']:.1f} <span class="text-xs font-normal text-gray-400">/ 100</span></td>
                <td class="py-3 px-4 border-b font-mono text-gray-600">${row['cost']:.4f}</td>
                <td class="py-3 px-4 border-b font-bold text-green-600">{row['bang_for_buck']:.3f}</td>
                <td class="py-3 px-4 border-b font-mono text-gray-600">{row['time_mins']:.1f}m</td>
                <td class="py-3 px-4 border-b font-bold text-blue-600">{row['velocity']:.4f}</td>
                <td class="py-3 px-4 border-b font-bold text-purple-600">{row['density']:.3f}</td>
            </tr>
        """
        
    html += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return html

######################################

import pandas as pd
import os

def get_resource_insights_data():
    """Reads and aggregates resource and score data from resource_metrics.csv across ALL tasks."""
    merged = []
    csv_path = "data/resource_metrics.csv" # Ensure this matches your file location
    
    if not os.path.exists(csv_path):
        print(f"Warning: Could not find {csv_path}!")
        return merged
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return merged
        
    # Clean the data: replace any empty cells with 0 to prevent math errors
    df = df.fillna(0)
        
    # Aggregate the data across all tasks (Single, Multi, and Cognitive)
    # Average the score, but SUM the costs, time, and tokens!
    agg_df = df.groupby('model_name').agg({
        'score': 'mean',
        'cost_usd': 'sum',
        'time_seconds': 'sum',
        'output_tokens': 'sum'
    }).reset_index()
    
    for _, row in agg_df.iterrows():
        model = str(row['model_name'])
        cost = float(row['cost_usd'])
        time_s = float(row['time_seconds'])
        tokens = float(row['output_tokens'])
        score = float(row['score'])
        
        merged.append({
            "model_name": model,
            "score": score,
            "cost": cost,
            "time_mins": time_s / 60.0,
            "tokens": tokens,
            # Safe math to prevent DivisionByZero crashes
            "score_per_dollar": score / cost if cost > 0 else 0,
            "score_per_min": score / (time_s / 60.0) if time_s > 0 else 0,
            "score_per_10k_tokens": (score / tokens) * 10000 if tokens > 0 else 0
        })
        
    return merged





def build_efficiency_kpis_html(data):
    if not data:
        return ""

    # Find the winners
    top_dollar = max(data, key=lambda x: x["score_per_dollar"])
    top_speed = max(data, key=lambda x: x["score_per_min"])
    top_dense = max(data, key=lambda x: x["score_per_10k_tokens"])

    html = f"""
    <div class="mb-12">
        <h3 class="text-2xl font-bold mb-4">Operational Excellence Highlights</h3>
        <div class="grid md:grid-cols-3 gap-6">
            
            <div class="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg p-6 shadow-sm">
                <div class="text-green-800 text-xs font-bold uppercase tracking-wider mb-1">Most Economical Genius</div>
                <div class="text-gray-600 text-sm mb-4">Highest Score per Dollar</div>
                <div class="text-2xl font-black text-gray-900 mb-1">{top_dollar['model_name']}</div>
                <div class="text-green-700 font-semibold">{top_dollar['score_per_dollar']:.3f} <span class="text-sm font-normal text-green-600">points/$</span></div>
            </div>

            <div class="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-6 shadow-sm">
                <div class="text-blue-800 text-xs font-bold uppercase tracking-wider mb-1">Fastest Decision Maker</div>
                <div class="text-gray-600 text-sm mb-4">Highest Score per Minute</div>
                <div class="text-2xl font-black text-gray-900 mb-1">{top_speed['model_name']}</div>
                <div class="text-blue-700 font-semibold">{top_speed['score_per_min']:.3f} <span class="text-sm font-normal text-blue-600">points/min</span></div>
            </div>

            <div class="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-6 shadow-sm">
                <div class="text-purple-800 text-xs font-bold uppercase tracking-wider mb-1">Most Concise Thinker</div>
                <div class="text-gray-600 text-sm mb-4">Highest Score per 10k Tokens</div>
                <div class="text-2xl font-black text-gray-900 mb-1">{top_dense['model_name']}</div>
                <div class="text-purple-700 font-semibold">{top_dense['score_per_10k_tokens']:.3f} <span class="text-sm font-normal text-purple-600">points/10k</span></div>
            </div>

        </div>
    </div>
    """
    return html





import statistics

import statistics

def build_efficiency_scatter_html(data):
    if not data:
        return ""

    # Calculate medians for the crosshairs
    median_cost = statistics.median([d["cost"] for d in data])
    median_score = statistics.median([d["score"] for d in data])

    # Prepare datasets colored by quadrant
    datasets = {
        "Value Champions (High Score, Low Cost)": {"color": "rgba(16, 185, 129, 1)", "data": []},
        "Premium Brains (High Score, High Cost)": {"color": "rgba(59, 130, 246, 1)", "data": []},
        "Lightweight (Low Score, Low Cost)": {"color": "rgba(156, 163, 175, 1)", "data": []},
        "Overpriced (Low Score, High Cost)": {"color": "rgba(239, 68, 68, 1)", "data": []},
    }

    for d in data:
        if d["score"] >= median_score and d["cost"] <= median_cost:
            cat = "Value Champions (High Score, Low Cost)"
        elif d["score"] >= median_score and d["cost"] > median_cost:
            cat = "Premium Brains (High Score, High Cost)"
        elif d["score"] < median_score and d["cost"] <= median_cost:
            cat = "Lightweight (Low Score, Low Cost)"
        else:
            cat = "Overpriced (Low Score, High Cost)"
            
        datasets[cat]["data"].append({
            "x": d["cost"],
            "y": d["score"],
            "model": d["model_name"]
        })

    # Generate Chart.js Dataset syntax
    js_datasets = []
    for label, info in datasets.items():
        if info["data"]:
            data_str = ", ".join([f"{{x: {v['x']}, y: {v['y']}, model: '{v['model']}'}}" for v in info["data"]])
            js_datasets.append(f"""
            {{
                label: '{label}',
                data: [{data_str}],
                backgroundColor: '{info["color"]}',
                borderColor: '{info["color"]}',
                pointRadius: 6,
                pointHoverRadius: 8
            }}
            """)

    html = f"""
    <div class="bg-white rounded-lg shadow p-6 border mb-12">
        <h3 class="text-2xl font-bold mb-2">Efficiency Quadrant: Brains vs. Wallet</h3>
        <p class="text-gray-600 mb-6 text-sm">Evaluating Multi-Turn cognitive performance against API cost. The ideal model sits in the top-left (Value Champions).</p>
        
        <div class="w-full relative h-[900px]">
            <canvas id="scatterChart"></canvas>
        </div>
    </div>
    
   # <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>
    
    <script>
    // Register the plugin
    Chart.register(ChartDataLabels);
    
    new Chart(document.getElementById('scatterChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{",".join(js_datasets)}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            layout: {{
                padding: {{ top: 20, right: 30 }} // Give labels room to breathe
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Total API Cost ($)', font: {{ weight: 'bold' }} }},
                    grid: {{ color: (ctx) => ctx.tick.value === {median_cost} ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.05)' }}
                }},
                y: {{
                    title: {{ display: true, text: 'Overall Executive Score', font: {{ weight: 'bold' }} }},
                    grid: {{ color: (ctx) => ctx.tick.value === {median_score} ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.05)' }}
                }}
            }},
            plugins: {{
                legend: {{ position: 'bottom', labels: {{ usePointStyle: true, padding: 20 }} }},
                // Configure the permanent labels
                datalabels: {{
                    align: 'top',
                    offset: 6,
                    color: '#4B5563', // Tailwind gray-600
                    font: {{ size: 10, weight: 'bold' }},
                    formatter: function(value, context) {{
                        return value.model;
                    }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            let item = context.raw;
                            return item.model + ': Score ' + item.y.toFixed(3) + ' | Cost $' + item.x.toFixed(2);
                        }}
                    }}
                }}
            }}
        }}
    }});
    </script>
    """
    return html


# ... (The rest of your Leaderboards section remains the same) ...


# --------------------------------------------------
# Leaderboards
# --------------------------------------------------

with open(LEADERBOARD_TEMPLATE, encoding="utf-8") as f:
    leaderboard_template = f.read()

task_cards = ""

 

# Load all resource data globally before the loop
resource_data = load_resource_metrics()

for task in TASKS:
    folder = os.path.join(DATA_ROOT, task["id"])
    results = load_json_results(folder)
    
    # 1. Merge resource data into the JSON results
    task_resources = resource_data.get(task["id"], {})
    for r in results:
        model_name = r["model_name"]
        res = task_resources.get(model_name, {})
        
        # Inject CSV data if it exists, otherwise default to 0
        r["input_tokens"] = res.get("input_tokens", 0)
        r["output_tokens"] = res.get("output_tokens", 0)
        r["cost_usd"] = res.get("cost_usd", 0.0)
        r["time_seconds"] = res.get("time_seconds", 0.0)

    # 2. Build the resource HTML section
    resource_section_html = build_efficiency_html(task_resources)

    # 3. Inject everything into the template
    html = (
        leaderboard_template
        .replace(TITLE_PLACEHOLDER, task["title"])
        .replace(DESCRIPTION_PLACEHOLDER, task["description"])
        .replace(ROW_PLACEHOLDER, build_table(results))
        .replace(NAV_PLACEHOLDER, navigation(task["id"]))
        .replace("{{RESOURCE_METRICS}}", resource_section_html) # NEW PLACEHOLDER
    )

    output = f"{task['id']}.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html) 

    task_cards += f"""

<div class="bg-white rounded-lg shadow p-6 border">
<h2 class="text-xl font-bold mb-2">{task['title']}</h2>
<p class="text-gray-600 mb-4">{task['description']}</p>
<p class="text-sm text-gray-500 mb-4">{len(results)} models evaluated</p>
<a class="text-blue-600 font-semibold hover:underline"
href="{task['id']}.html">
View Leaderboard →
</a>
</div>
"""

# --------------------------------------------------
# Home page (Updated to include ALL insights)
# --------------------------------------------------

# 1. Degradation
degradation_data = calculate_cognitive_degradation()
degradation_html = build_degradation_html(degradation_data)

# 2. Strategy
strategy_data = calculate_strategy_quadrant()
strategy_html = build_strategy_html(strategy_data)

# 3. Compliance
compliance_data = calculate_compliance_score()
compliance_html = build_compliance_html(compliance_data)

# 4. Profiles
profile_data = calculate_executive_profiles()
profile_html = build_executive_profile_html(profile_data)

# 5. Pillars

pillars_data = calculate_cognitive_pillars()
pillars_html = build_pillars_html(pillars_data)


# 6. Calculate the new combined insights

combined_df = get_weighted_scores()
combined_bar_html = build_combined_bar_chart_html(combined_df) 
 


#7. New Efficiency Insights ---
resource_insights_data = get_resource_insights_data()
kpi_html = build_efficiency_kpis_html(resource_insights_data)
scatter_html = build_efficiency_scatter_html(resource_insights_data)








# ... (Standard replacement and save logic)



with open(HOME_TEMPLATE, encoding="utf-8") as f:
    home = f.read()

# Combine all insights in a logical storytelling order 

# Combine all insights in a logical storytelling order 

combined_insights = (
    kpi_html + 
    combined_bar_html + 
    scatter_html + 
    degradation_html + 
    pillars_html +  
    strategy_html + 
    compliance_html
)



#combined_insights = combined_bar_html + degradation_html + pillars_html +  strategy_html + compliance_html

home = (
    home
    .replace(CARDS_PLACEHOLDER, task_cards)
    .replace(NAV_PLACEHOLDER, navigation("home"))
    .replace(COGNITIVE_DEGRADATION_PLACEHOLDER, combined_insights) 
)

with open(HOME_OUTPUT, "w", encoding="utf-8") as f:
    f.write(home)

print("Website generated successfully with the Executive Profiles!")


