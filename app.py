import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pathlib import Path
import textwrap
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Football Match Dynamics Dashboard",
    layout="wide"
)

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e1065, #581c87, #7e22ce);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.kpi-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #334155;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.35);
    margin-bottom: 15px;
}

.kpi-title {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 34px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

.kpi-note {
    font-size: 13px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("⚽ Football Match Dynamics Dashboard")
st.write("Interactive analysis of football team shape using tracking data.")

with st.expander("About this Dashboard", expanded=False):
    st.markdown("""
    This dashboard analyses football match dynamics using player tracking data.

    It visualises team shape and spatial organisation through:
    - Width
    - Depth
    - Compactness
    - Convex Hull Area

    It also includes team comparison, player position analysis, and tactical insights.
    """)

# -----------------------------
# Dataset paths
# -----------------------------
DATA_DIR = Path("data")

match_files = {
    "Netherlands vs Senegal": DATA_DIR / "netherlands_senegal.txt",
    "France vs Argentina": DATA_DIR / "france_argentina.txt",
    "Germany vs Spain": DATA_DIR / "germany_spain.txt"
}

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Navigation")

selected_match = st.sidebar.selectbox(
    "Select Match",
    list(match_files.keys())
)

df = load_data(str(match_files[selected_match]))

metric_choice = st.sidebar.selectbox(
    "Select Metric",
    ["width", "depth", "compactness", "area"]
)

half_filter = st.sidebar.selectbox(
    "Select Match Period",
    ["Full Match", "First Half", "Second Half"]
)

if half_filter == "First Half":
    df = df[df["period"] == 1]

elif half_filter == "Second Half":
    df = df[df["period"] == 2]

team_filter = st.sidebar.selectbox(
    "Select Team View",
    ["Both Teams"] + sorted(df["team_name"].unique().tolist())
)

sample_step = st.sidebar.slider(
    "Graph Detail Level",
    min_value=1,
    max_value=20,
    value=5,
    help="1 gives full detail but loads slower. Higher values load faster."
)

st.sidebar.caption(
    "Tip: 5 is balanced. Use 1 for full detail or 10–20 for faster loading."
)

# -----------------------------
# Metric labels
# -----------------------------
metric_titles = {
    "width": "Team Width Over Time",
    "depth": "Team Depth Over Time",
    "compactness": "Team Compactness Over Time",
    "area": "Convex Hull Area Over Time"
}

metric_units = {
    "width": "m",
    "depth": "m",
    "compactness": "m",
    "area": "m²"
}

# -----------------------------
# Compute metric
# -----------------------------
@st.cache_data
def compute_metric(df, metric_name, sample_step):
    timestamps = np.sort(df["timestamp_s"].unique())
    sampled_timestamps = timestamps[::sample_step]
    df_sampled = df[df["timestamp_s"].isin(sampled_timestamps)].copy()

    results = []

    for (timestamp, team), group in df_sampled.groupby(["timestamp_s", "team_name"]):
        centroid_x = group["x"].mean()
        centroid_y = group["y"].mean()

        if metric_name == "width":
            value = group["x"].max() - group["x"].min()

        elif metric_name == "depth":
            value = group["y"].max() - group["y"].min()

        elif metric_name == "compactness":
            distances = np.sqrt(
                (group["x"] - centroid_x) ** 2 +
                (group["y"] - centroid_y) ** 2
            )
            value = distances.mean()

        elif metric_name == "area":
            points = group[["x", "y"]].dropna().values

            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    value = hull.volume
                except Exception:
                    value = np.nan
            else:
                value = np.nan

        results.append({
            "timestamp_s": timestamp,
            "team_name": team,
            metric_name: value
        })

    return pd.DataFrame(results)

metrics = compute_metric(df, metric_choice, sample_step)

if team_filter != "Both Teams":
    metrics = metrics[metrics["team_name"] == team_filter]

# -----------------------------
# Main heading
# -----------------------------
st.subheader(f"🏟️ {selected_match}")



# -----------------------------
# Dataset preview
# -----------------------------
with st.expander("Dataset Preview", expanded=False):
    important_columns = [
        "timestamp_s",
        "team_name",
        "player_name",
        "x",
        "y",
        "speed"
    ]
    st.dataframe(df[important_columns].head())

# -----------------------------
# Metric explanation
# -----------------------------
with st.expander("What do these metrics mean?", expanded=False):
    st.markdown("""
    **Width**: how spread out a team is horizontally.  
    **Depth**: how stretched a team is from front to back.  
    **Compactness**: how close players are to the team centroid.  
    **Convex Hull Area**: total pitch space occupied by the team shape.
    """)

# -----------------------------
# Team Summary Statistics
# -----------------------------
st.subheader("📊 Team Summary Statistics")

summary_stats = (
    metrics.groupby("team_name")[metric_choice]
    .mean()
    .reset_index()
)

team_info = {
    "Netherlands": {"flag": "https://flagcdn.com/w80/nl.png", "association": "KNVB", "color": "#2563eb"},
    "Senegal": {"flag": "https://flagcdn.com/w80/sn.png", "association": "FSF", "color": "#16a34a"},
    "France": {"flag": "https://flagcdn.com/w80/fr.png", "association": "FFF", "color": "#2563eb"},
    "Argentina": {"flag": "https://flagcdn.com/w80/ar.png", "association": "AFA", "color": "#38bdf8"},
    "Germany": {"flag": "https://flagcdn.com/w80/de.png", "association": "DFB", "color": "#111827"},
    "Spain": {"flag": "https://flagcdn.com/w80/es.png", "association": "RFEF", "color": "#dc2626"}
}

cols = st.columns(len(summary_stats))

for i, row in summary_stats.iterrows():
    team = row["team_name"]
    

    team_df = df[df["team_name"] == team]
    players_tracked = team_df["player_name"].nunique()
    avg_speed = team_df["speed"].mean()

    total_seconds = team_df["timestamp_s"].max() - team_df["timestamp_s"].min()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    info = team_info.get(team, {
        "flag": "https://flagcdn.com/w80/un.png",
        "association": "Team",
        "color": "#7e22ce"
    })

    flag_url = info["flag"]
    association = info["association"]
    team_color = info["color"]

    html_card = textwrap.dedent(f"""
    <div style="
        background:white;
        color:#111827;
        padding:14px 18px;
        border-radius:18px;
        border:1px solid #e5e7eb;
        box-shadow:0px 4px 16px rgba(0,0,0,0.12);
        margin-bottom:4px;
    ">

        <div style="display:flex; justify-content:space-between; align-items:center;">

    <div style="display:flex; gap:16px; align-items:center;">
        <img src="{flag_url}" width="46" height="46"
        style="border-radius:50%; object-fit:cover;">

        <div>
            <div style="font-size:22px; font-weight:800; color:{team_color};">
                {team}
            </div>

            <div style="font-size:15px; color:#374151;">
                {association}
            </div>
        </div>
    </div>

</div>

            
        

        <hr style="border:none; border-top:1px solid #e5e7eb; margin:10px 0;">

        <div style="display:flex; justify-content:space-between; text-align:center;">
            <div>
                <div style="font-size:17px; color:#6b7280;">Players Tracked</div>
                <div style="font-size:17px; font-weight:800;">{players_tracked}</div>
            </div>

            <div>
                <div style="font-size:17px; color:#6b7280;">Total Time</div>
                <div style="font-size:17px; font-weight:800;">{minutes}:{seconds:02d}</div>
            </div>

            <div>
                <div style="font-size:17px; color:#6b7280;">Avg Team Speed</div>
                <div style="font-size:17px; font-weight:800; color:{team_color};">
                    {avg_speed:.2f} m/s
                </div>
            </div>
        </div>
    </div>
    """)

    with cols[i]:
        components.html(html_card, height=170)
        

    
        
# -----------------------------
# Automatic Match Insights
# -----------------------------
if len(summary_stats) >= 2:
    st.subheader("🧠 Match Insights")

    team_metrics = metrics.groupby("team_name")[metric_choice].mean()

    best_team = team_metrics.idxmax()
    worst_team = team_metrics.idxmin()

    best_value = team_metrics.max()
    worst_value = team_metrics.min()

    difference = best_value - worst_value

    insight_1 = (
        f"{best_team} maintained a higher average "
        f"{metric_choice} throughout the match."
    )

    insight_2 = (
        f"The difference between teams was "
        f"{difference:.2f}{metric_units[metric_choice]}."
    )

    if metric_choice == "width":
        insight_3 = f"{best_team} displayed a wider tactical structure."

    elif metric_choice == "depth":
        insight_3 = f"{best_team} maintained greater vertical spacing."

    elif metric_choice == "compactness":
        insight_3 = f"{best_team} showed stronger team compactness."

    elif metric_choice == "area":
        insight_3 = f"{best_team} occupied a larger pitch area."

    else:
        insight_3 = f"{best_team} showed stronger values for this metric."

    st.markdown(
        f"""
        <div style="
            background-color:#111827;
            padding:20px;
            border-radius:12px;
            border-left:6px solid #3b82f6;
            margin-bottom:25px;
        ">
        <h4 style="color:white;">Key Tactical Insights</h4>
        <ul style="color:#d1d5db; font-size:16px;">
            <li>{insight_1}</li>
            <li>{insight_2}</li>
            <li>{insight_3}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Smoothed Line Chart
# -----------------------------
metrics["smoothed_metric"] = (
    metrics.groupby("team_name")[metric_choice]
    .transform(lambda x: x.rolling(window=120, min_periods=1).mean())
)

st.subheader(f"📈 {metric_titles[metric_choice]}")

line_fig = px.line(
    metrics,
    x="timestamp_s",
    y="smoothed_metric",
    color="team_name",
    color_discrete_sequence=["#60a5fa", "#22c55e"],
    labels={
        "timestamp_s": "Time (seconds)",
        "smoothed_metric": f"{metric_choice.title()} ({metric_units[metric_choice]})",
        "team_name": "Team"
    }
)

line_fig.update_traces(line=dict(width=2))

line_fig.update_layout(
    title="",
    height=500,
    plot_bgcolor="#f8fafc",
paper_bgcolor="#f8fafc",
font=dict(color="#111827"),
    margin=dict(l=50, r=25, t=25, b=50),

    xaxis=dict(
        title="Time (seconds)",
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=13),
        gridcolor="#e5e7eb",
        showline=True,
        linewidth=2,
        linecolor="#374151",
        mirror=True
    ),

    yaxis=dict(
        title=f"{metric_choice.title()} ({metric_units[metric_choice]})",
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=13),
        showgrid=True,
        gridcolor="#d3d3d3",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    ),

    legend=dict(
        title="Team",
        title_font=dict(color="black", size=14),
        font=dict(color="black", size=12),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="gray",
        borderwidth=1
    )
)

st.plotly_chart(line_fig, use_container_width=True)

# -----------------------------
# Team Comparison Bar Chart
# -----------------------------
st.subheader("⚽ Team Comparison")

bar_fig = px.bar(
    summary_stats,
    x="team_name",
    y=metric_choice,
    color="team_name",
    color_discrete_sequence=["#60a5fa", "#22c55e"],
    labels={
        "team_name": "Team",
        metric_choice: f"Average {metric_choice.title()} ({metric_units[metric_choice]})"
    }
)

bar_fig.update_layout(
    height=500,
    title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", size=15),
    showlegend=True,
    margin=dict(l=30, r=20, t=25, b=35),

    xaxis=dict(
        title="Team",
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=13),
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    ),

    yaxis=dict(
        title=f"Average {metric_choice.title()} ({metric_units[metric_choice]})",
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=13),
        showgrid=True,
        gridcolor="#d3d3d3",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    ),

    legend=dict(
        title="Team",
        title_font=dict(color="black", size=14),
        font=dict(color="black", size=12),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="gray",
        borderwidth=1
    )
)

st.plotly_chart(bar_fig, use_container_width=True)

# -----------------------------
# Metric Summary Table
# -----------------------------
st.subheader("📋 Metric Summary Table")

summary_display = summary_stats.copy()
summary_display[metric_choice] = summary_display[metric_choice].round(2)

st.dataframe(summary_display, use_container_width=True)

csv_summary = summary_display.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Metric Summary CSV",
    data=csv_summary,
    file_name=f"{selected_match}_{metric_choice}_summary.csv",
    mime="text/csv"
)



# -----------------------------
# Pitch Visualisation
# -----------------------------
st.subheader("⚽ Player Position Snapshot")

timestamps = np.sort(df["timestamp_s"].unique())

# -----------------------------
# Playback controls
# -----------------------------
play_animation = st.checkbox("▶ Play Animation", value=False)

animation_speed = st.slider(
    "Animation Speed (milliseconds)",
    min_value=200,
    max_value=2000,
    value=800,
    step=100
)

if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0

if play_animation:
    st_autorefresh(interval=animation_speed, key="pitch_animation")

    st.session_state.frame_index += 30

    if st.session_state.frame_index >= len(timestamps):
        st.session_state.frame_index = 0

    selected_timestamp = float(timestamps[st.session_state.frame_index])

else:
    selected_timestamp = st.slider(
        "Select Timestamp",
        min_value=float(timestamps.min()),
        max_value=float(timestamps.max()),
        value=float(timestamps[0])
    )

    st.session_state.frame_index = np.abs(
        timestamps - selected_timestamp
    ).argmin()

nearest_timestamp = timestamps[
    np.abs(timestamps - selected_timestamp).argmin()
]

frame = df[df["timestamp_s"] == nearest_timestamp]

trail_frames = df[
    (df["timestamp_s"] >= nearest_timestamp - 5) &
    (df["timestamp_s"] <= nearest_timestamp)
]

if team_filter != "Both Teams":
    frame = frame[frame["team_name"] == team_filter]
    trail_frames = trail_frames[trail_frames["team_name"] == team_filter]

st.write(
    f"Showing player positions at timestamp: "
    f"{nearest_timestamp:.2f} seconds"
)

pitch_fig = px.scatter(
    frame,
    x="x",
    y="y",
    color="team_name",
    hover_data=["player_name", "team_name", "speed"],
    color_discrete_sequence=["#60a5fa", "#22c55e"],
    labels={
        "x": "X Position",
        "y": "Y Position",
        "team_name": "Team"
    }
)

# Movement trails
for player, player_group in trail_frames.groupby("player_name"):
    player_group = player_group.sort_values("timestamp_s")

    pitch_fig.add_trace(
        go.Scatter(
            x=player_group["x"],
            y=player_group["y"],
            mode="lines",
            line=dict(width=1, dash="dot"),
            opacity=0.35,
            showlegend=False,
            hoverinfo="skip"
        )
    )

# Centroids and convex hulls
for team, group in frame.groupby("team_name"):
    centroid_x = group["x"].mean()
    centroid_y = group["y"].mean()

    pitch_fig.add_trace(
        go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode="markers",
            marker=dict(size=16, symbol="x", color="red"),
            name=f"{team} Centroid"
        )
    )

    points = group[["x", "y"]].dropna().values

    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            hull_x = list(hull_points[:, 0]) + [hull_points[0, 0]]
            hull_y = list(hull_points[:, 1]) + [hull_points[0, 1]]

            pitch_fig.add_trace(
                go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode="lines",
                    line=dict(width=2),
                    name=f"{team} Shape"
                )
            )

        except Exception:
            pass

# Pitch markings
pitch_fig.add_shape(type="rect", x0=-52.5, y0=-34, x1=52.5, y1=34, line=dict(color="white", width=3))
pitch_fig.add_shape(type="line", x0=0, y0=-34, x1=0, y1=34, line=dict(color="white", width=2))
pitch_fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=dict(color="white", width=2))
pitch_fig.add_shape(type="rect", x0=-52.5, y0=-20.15, x1=-36, y1=20.15, line=dict(color="white", width=2))
pitch_fig.add_shape(type="rect", x0=36, y0=-20.15, x1=52.5, y1=20.15, line=dict(color="white", width=2))
pitch_fig.add_shape(type="rect", x0=-52.5, y0=-9.16, x1=-47, y1=9.16, line=dict(color="white", width=2))
pitch_fig.add_shape(type="rect", x0=47, y0=-9.16, x1=52.5, y1=9.16, line=dict(color="white", width=2))
pitch_fig.add_shape(type="circle", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5, fillcolor="white", line=dict(color="white"))

pitch_fig.update_layout(
    height=500,
    title=None,
    plot_bgcolor="#7bbf6a",
    paper_bgcolor="white",
    font=dict(color="black", size=15),
    margin=dict(l=20, r=20, t=25, b=25),

    xaxis=dict(
        range=[-55, 55],
        showgrid=False,
        zeroline=False,
        visible=False
    ),

    yaxis=dict(
        range=[-36, 36],
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1
    ),

    legend=dict(
        title="Team",
        title_font=dict(color="black", size=14),
        font=dict(color="black", size=12),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="gray",
        borderwidth=1
    )
)

st.plotly_chart(pitch_fig, use_container_width=True)

# -----------------------------
# Cross-Match Comparison
# -----------------------------
st.subheader("📈 Cross-Match Comparison")

@st.cache_data
def compute_all_match_summaries(match_files, sample_step):
    all_summaries = []

    for match_name, file_path in match_files.items():
        df_match = pd.read_csv(file_path)

        for metric in ["width", "depth", "compactness", "area"]:
            metric_df = compute_metric(df_match, metric, sample_step)

            summary = (
                metric_df.groupby("team_name")[metric]
                .mean()
                .reset_index()
            )

            summary["match"] = match_name
            summary["metric"] = metric
            summary["value"] = summary[metric]

            all_summaries.append(
                summary[["match", "team_name", "metric", "value"]]
            )

    return pd.concat(all_summaries, ignore_index=True)

comparison_df = compute_all_match_summaries(match_files, sample_step)

comparison_metric = st.selectbox(
    "Select Metric for Cross-Match Comparison",
    ["width", "depth", "compactness", "area"]
)

filtered_comparison = comparison_df[
    comparison_df["metric"] == comparison_metric
]

comparison_fig = px.bar(
    filtered_comparison,
    x="team_name",
    y="value",
    color="match",
    barmode="group",
    labels={
        "team_name": "Team",
        "value": f"Average {comparison_metric.title()} ({metric_units[comparison_metric]})",
        "match": "Match"
    },
    color_discrete_sequence=px.colors.qualitative.Set2
)

comparison_fig.update_layout(
    height=500,
    title="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", size=15),
    margin=dict(l=30, r=20, t=25, b=35),

    xaxis=dict(
        title="Team",
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=13),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    ),

    yaxis=dict(
        title=f"Average {comparison_metric.title()} ({metric_units[comparison_metric]})",
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=13),
        showgrid=True,
        gridcolor="#d3d3d3",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    ),

    legend=dict(
        title="Match",
        title_font=dict(color="black", size=14),
        font=dict(color="black", size=12),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="gray",
        borderwidth=1
    )
)

st.plotly_chart(comparison_fig, use_container_width=True)

st.dataframe(
    filtered_comparison.round(2),
    use_container_width=True
)