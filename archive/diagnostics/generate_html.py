import pandas as pd
import json
import os

files_to_plot = [
    "realKnownCause/sixeyes_zscore_nyc_taxi.csv",
    "realKnownCause/sixeyes_zscore_machine_temperature_system_failure.csv"
]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "nab"))
labels_file = os.path.join(base_dir, "labels", "combined_windows.json")

with open(labels_file, "r") as f:
    windows_dict = json.load(f)

html = """
<html>
<head>
    <title>Diagnostic Charts</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .chart-container { width: 90%; height: 400px; margin: auto; margin-bottom: 50px; }
    </style>
</head>
<body>
    <h2>Z-Score Diagnostic Charts</h2>
"""

for fpath in files_to_plot:
    rel_path = fpath.replace("sixeyes_zscore_", "")
    zscore_csv = os.path.join(base_dir, "results", "sixeyes_zscore", fpath)
    
    if not os.path.exists(zscore_csv):
        continue
        
    df = pd.read_csv(zscore_csv)
    windows = windows_dict.get(rel_path, [])
    
    # Subsample data if too large, but for chart.js ~5000 points is ok
    labels = df['timestamp'].tolist()
    values = df['value'].tolist()
    scores = df['anomaly_score'].tolist()
    
    # Create anomaly window background boxes
    annotations = []
    for w in windows:
        annotations.append(f"{{ type: 'box', xMin: '{w[0]}', xMax: '{w[1]}', backgroundColor: 'rgba(255, 99, 132, 0.25)', borderWidth: 0 }}")
    
    chart_id = "chart_" + fpath.replace("/", "_").replace(".", "_")
    
    html += f"""
    <div class="chart-container">
        <canvas id="{chart_id}"></canvas>
    </div>
    <script>
    new Chart(document.getElementById('{chart_id}'), {{
        type: 'line',
        data: {{
            labels: {json.dumps(labels)},
            datasets: [
                {{
                    label: 'Value (normalized)',
                    data: {json.dumps([(v - min(values))/(max(values)-min(values)) for v in values])},
                    borderColor: 'blue',
                    borderWidth: 1,
                    pointRadius: 0
                }},
                {{
                    label: 'Anomaly Score (Z-Score)',
                    data: {json.dumps(scores)},
                    borderColor: 'red',
                    borderWidth: 1.5,
                    pointRadius: 0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: '{rel_path}' }}
            }}
        }}
    }});
    </script>
    """

html += "</body></html>"

with open("diagnostic_charts.html", "w") as f:
    f.write(html)
print("Generado diagnostic_charts.html")
