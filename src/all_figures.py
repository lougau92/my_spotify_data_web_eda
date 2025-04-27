import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Style configuration
color_palette = px.colors.qualitative.Plotly
template = "plotly_white"
font_family = "Arial"
title_font = dict(size=24, family=font_family, color="#2a3f5f")
axis_title_font = dict(size=18, family=font_family, color="#2a3f5f")
tick_font = dict(size=14, family=font_family, color="#636363")
hoverlabel = dict(bgcolor="white", font_size=14, font_family=font_family)

# Common layout parameters
common_layout = dict(
    template=template,
    font=dict(family=font_family),
    hoverlabel=hoverlabel,
    title_font=title_font,
    # xaxis=dict(title_font=axis_title_font, tickfont=tick_font),
    # yaxis=dict(title_font=axis_title_font, tickfont=tick_font),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=50, b=50, t=80, pad=10),
)

# Distribution of Song Play Counts: Percentage of songs by how many times they were played
def create_play_count_distribution(song_play_counts):
# Create bins: 1, 2, 3, 4, 5, 6+
    song_play_counts["play_count_grouped"] = song_play_counts["play_count"].apply(
        lambda x: f"6+" if x >= 6 else str(x)
    )

    # Calculate distribution
    grouped_distribution = (
        song_play_counts["play_count_grouped"].value_counts().sort_index()
    )
    grouped_distribution_pct = (grouped_distribution / grouped_distribution.sum()) * 100

    # Create ordered categories
    categories = ["1", "2", "3", "4", "5", "6+"]

    # Create DataFrame for plotting
    dist_df = pd.DataFrame(
        {
            "play_count": pd.Categorical(
                categories, categories=categories, ordered=True
            ),
            "percentage": [grouped_distribution_pct.get(cat, 0) for cat in categories],
        }
    )

    # Create the figure
    fig = px.bar(
        dist_df,
        x="play_count",
        y="percentage",
        title="<b>Distribution of Song Play Counts</b><br><sup>Percentage of songs by how many times they were played</sup>",
        labels={"play_count": "Number of Plays", "percentage": "Percentage of Songs"},
        color="play_count",
        color_discrete_sequence=px.colors.sequential.Viridis,
        category_orders={"play_count": categories},
    )

    # Update traces for better styling
    fig.update_traces(
        hovertemplate="<b>%{x} plays</b><br>%{y:.2f}% of songs<extra></extra>",
        marker_line_width=0,
        texttemplate="%{y:.1f}%",
        textposition="outside",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        showlegend=False,
        xaxis=dict(
            title="<b>Number of Plays</b>",
            type="category",
            range=[
                -0.5,
                len(categories) - 0.5,
            ],  # Ensures x-axis starts at first category
        ),
        yaxis=dict(
            title="<b>Percentage of Songs</b>",
            ticksuffix="%",
            range=[
                0,
                max(grouped_distribution_pct) * 1.1,
            ],  # Starts at 0 with 10% padding
        ),
        hovermode="x unified",
    )

    return fig

# Song Duration Distribution How long are the songs you listen to?
def create_duration_distribution(df_songs):
# Convert to minutes for more intuitive interpretation
    duration_min = df_songs["secondsPlayed"] / 60

    fig = px.histogram(
        x=duration_min,
        nbins=35,
        title="<b>Song Duration Distribution</b><br><sup>How long are the songs you listen to?</sup>",
        labels={"x": "Duration (minutes)", "count": "Number of Songs"},
        color_discrete_sequence=[color_palette[0]],
    )

    # Update traces for better styling
    fig.update_traces(
        hovertemplate="<b>%{x:.1f} mins</b><br>Count: %{y}",
        marker_line_width=0.5,
        marker_line_color="white",
        opacity=0.8,
        xbins=dict(size=0.5),  # Bin size of 0.5 minutes (30 seconds)
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        bargap=0.05,
        xaxis=dict(
            title="<b>Duration (minutes)</b>",
            range=[0, duration_min.max()],  # Start at 0
            tickmode="linear",
            dtick=1,  # Show tick every minute
        ),
        yaxis=dict(title="<b>Number of Songs</b>"),
        hovermode="x",
    )

    # Add vertical line at median duration
    median_duration = duration_min.median()
    fig.add_vline(
        x=median_duration,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Median: {median_duration:.1f} mins",
        annotation_position="top right",
        annotation_font_size=12,
    )

    return fig

# Top {n} Most Played Artists
def create_top_artists_bar(df_songs, top_artists, n):
 # Create consistent color mapping for artists
    artist_colors = {
        artist: color_palette[i % len(color_palette)]
        for i, artist in enumerate(
            df_songs["artistName"].value_counts().nlargest(50).index
        )
    }

    fig = px.bar(
        top_artists.reset_index(),
        x="count",
        y="artistName",
        color="artistName",
        color_discrete_map=artist_colors,
        title=f"<b>Top {n} Most Played Artists</b>",
        labels={"count": "Number of Plays", "artistName": "Artist"},
        orientation="h",
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x} plays<extra></extra>", marker_line_width=0
    )

    fig.update_layout(
        **common_layout,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="<b>Number of Plays</b>",
        yaxis_title="<b>Artist</b>",
        height=600,  # Increased height for better artist name visibility
    )

    return fig, artist_colors  # Return color mapping for reuse

# Most Played Artist Per Month
def create_top_artist_monthly(artist_monthly, artist_colors):
 # Get top artist per month
    top_artists_monthly = artist_monthly.loc[
        artist_monthly.groupby("month_year")["count"].idxmax()
    ]

    # Create plot with consistent colors
    fig = px.bar(
        top_artists_monthly,
        x="month_year",
        y="count",
        color="artistName",
        color_discrete_map=artist_colors,
        title="<b>Most Played Artist Per Month</b>",
        labels={"count": "Number of Plays", "month_year": "Month"},
        category_orders={"month_year": sorted(artist_monthly["month_year"].unique())},
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Top Artist: %{fullData.name}<br>Plays: %{y}<extra></extra>",
        marker_line_width=0,
    )

    fig.update_layout(
        **common_layout,
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Number of Plays</b>",
        legend_title="Artist",
        bargap=0.2
    )

    return fig

# Listening Patterns of Top Artists Over Time: Weekly aggregated plays
def create_artist_topic_river(df_artist_plays, artist_colors):
# Prepare data for plotting
    df_plot = df_artist_plays.reset_index().melt(
        id_vars="date", var_name="artist", value_name="plays"
    )

    fig = px.area(
        df_plot,
        x="date",
        y="plays",
        color="artist",
        color_discrete_map=artist_colors,
        title="<b>Listening Patterns of Top Artists Over Time</b><br><sup>Weekly aggregated plays</sup>",
        labels={"plays": "Number of Plays", "date": "Date"},
        facet_col=None,
    )

    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%b %d, %Y}<br>Plays: %{y}<extra></extra>",
        line_width=0.5,
        opacity=0.8,
    )

    fig.update_layout(
        **common_layout,
        xaxis_title="<b>Date</b>",
        yaxis_title="<b>Number of Plays</b>",
        legend_title="Artist",
        hovermode="x unified"
    )

    return fig

# Top {n} Most Played Tracks
def create_top_track_bar(df_songs, top_tracks, n):
# Create consistent color mapping for artists
    track_colors = {
        artist: color_palette[i % len(color_palette)]
        for i, artist in enumerate(
            df_songs["trackName"].value_counts().nlargest(50).index
        )
    }

    fig = px.bar(
        top_tracks.reset_index(),
        x="count",
        y="trackName",
        color="trackName",
        color_discrete_map=track_colors,
        title=f"<b>Top {n} Most Played Tracks</b>",
        labels={"count": "Number of Plays", "trackName": "Track"},
        orientation="h",
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x} plays<extra></extra>", marker_line_width=0
    )

    fig.update_layout(
        **common_layout,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="<b>Number of Plays</b>",
        yaxis_title="<b>Track</b>",
        height=600,
    )

    return fig, track_colors

# Most Played Track Per Month
def create_top_track_monthly(track_monthly, track_colors):
# Get top track per month
    top_tracks_monthly = track_monthly.loc[
        track_monthly.groupby("month_year")["count"].idxmax()
    ]

    # Create plot with consistent colors
    fig = px.bar(
        top_tracks_monthly,
        x="month_year",
        y="count",
        color="trackName",
        color_discrete_map=track_colors,
        title="<b>Most Played Track Per Month</b>",
        labels={"count": "Number of Plays", "month_year": "Month"},
        category_orders={"month_year": sorted(track_monthly["month_year"].unique())},
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Top Track: %{fullData.name}<br>Plays: %{y}<extra></extra>",
        marker_line_width=0,
    )

    fig.update_layout(
        **common_layout,
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Number of Plays</b>",
        legend_title="Track",
        bargap=0.2
    )

    return fig

# Listening Patterns of Top Tracks Over Time: Weekly aggregated plays
def create_artist_topic_river(df_track_plays, track_colors):
# Prepare data for plotting
    df_plot = df_track_plays.reset_index().melt(
        id_vars="date", var_name="track", value_name="plays"
    )

    fig = px.area(
        df_plot,
        x="date",
        y="plays",
        color="track",
        color_discrete_map=track_colors,
        title="<b>Listening Patterns of Top Tracks Over Time</b><br><sup>Weekly aggregated plays</sup>",
        labels={"plays": "Number of Plays", "date": "Date"},
        facet_col=None,
    )

    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%b %d, %Y}<br>Plays: %{y}<extra></extra>",
        line_width=0.5,
        opacity=0.8,
    )

    fig.update_layout(
        **common_layout,
        xaxis_title="<b>Date</b>",
        yaxis_title="<b>Number of Plays</b>",
        legend_title="Track",
        hovermode="x unified"
    )

    return fig

# Yearly Listening Distribution: Percentage of total listening time by year 
def create_yearly_listening_pie(yearly_listening):
 # Calculate percentage for annotation
    total_hours = yearly_listening["hours_played"].sum()

    fig = px.pie(
        yearly_listening,
        values="hours_played",
        names="year",
        title="<b>Yearly Listening Distribution</b><br><sup>Percentage of total listening time by year</sup>",
        hole=0.5,
        color_discrete_sequence=color_palette,
        labels={"hours_played": "Hours Played", "year": "Year"},
    )

    # Update traces for better readability
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=14,
        hovertemplate="<b>%{label}</b><br>%{value:.1f} hours (%{percent})<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
    )

    # Add annotation for total hours
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=f"Total: {total_hours:.0f} hours",
        showarrow=False,
        font=dict(size=16, color="#2a3f5f"),
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        uniformtext_minsize=12,  # Minimum font size for labels
        uniformtext_mode="hide",  # Hide labels that don't fit
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    return fig

# Daily Listening hours and a 30-day moving average
def create_daily_listening_trends(daily_listening):
 # Create the figure with modified 15-day average opacity
    fig = px.line(
        daily_listening,
        x="date",
        y=["hours_played", "30_day_avg"],
        title="<b>Daily Listening hours and a 30-day moving average </b>",
        labels={"value": "Listening Hours", "date": "Date"},
        color_discrete_sequence=[
            f"rgba{(*px.colors.hex_to_rgb(color_palette[0]), 0.7)}",  # 15-day avg with 70% opacity
            color_palette[1],  # 30-day avg stays solid
        ],
    )

    # Update traces separately for finer control
    fig.update_traces(
        selector={"name": "hours_played"},
        line=dict(width=2.5),
        opacity=0.7,  # Additional opacity control
        hovertemplate="<b>Daily listening</b><br>Date: %{x|%b %d, %Y}<br>Hours: %{y:.2f}<extra></extra>",
    )

    fig.update_traces(
        selector={"name": "30_day_avg"},
        line=dict(width=2.5),
        hovertemplate="<b>30-day avg</b><br>Date: %{x|%b %d, %Y}<br>Hours: %{y:.2f}<extra></extra>",
    )

    # Rest of the layout updates (same as before)
    fig.update_layout(
        **common_layout,
        hovermode="x unified",
        xaxis=dict(
            title="<b>Date</b>",
            tickformat="%b %Y",
            tickangle=45,
            # rangeslider=dict(visible=True),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="<b>Listening Hours</b>", gridcolor="lightgrey", gridwidth=0.5
        ),
        legend=dict(
            title="Moving Averages:",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Add annotations (same as before)
    max_val = daily_listening["hours_played"].max()
    max_date = daily_listening.loc[daily_listening["hours_played"].idxmax(), "date"]

    fig.add_annotation(
        x=max_date,
        y=max_val,
        text=f"Peak: {max_val:.1f} hours",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
    )

    avg_hours = daily_listening["hours_played"].mean()
    fig.add_hline(
        y=avg_hours,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Average: {avg_hours:.1f} hours",
        annotation_position="bottom right",
    )

    return fig

# Monthly Listening Trends by Year: Patterns across different months    
def create_monthly_trends_plot(monthly_listening, month_order):
# Create the figure with consistent color palette
    fig = px.line(
        monthly_listening,
        x="month",
        y="hours_played",
        color="year",
        title="<b>Monthly Listening Trends by Year</b><br><sup>Patterns across different months</sup>",
        labels={"hours_played": "Listening Hours", "month": "Month"},
        category_orders={"month": month_order},
        color_discrete_sequence=color_palette,
        markers=True,
    )

    # Update traces for better visualization
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=8),
        hovertemplate="<b>%{fullData.name}</b><br>Month: %{x}<br>Hours: %{y:.1f}<extra></extra>",
        mode="lines+markers",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        xaxis={
            "type": "category",
            "categoryorder": "array",
            "categoryarray": month_order,
            "title": "<b>Month</b>",
            "tickangle": -45,
        },
        yaxis={
            "title": "<b>Listening Hours</b>",
            "gridcolor": "lightgrey",
            "gridwidth": 0.5,
        },
        legend={
            "title": "Year:",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
    )

    # Add average reference line
    avg_hours = monthly_listening["hours_played"].mean()
    fig.add_hline(
        y=avg_hours,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Overall Avg: {avg_hours:.1f} hours",
        annotation_position="bottom right",
    )

    # Add peak month annotation if desired
    max_entry = monthly_listening.loc[monthly_listening["hours_played"].idxmax()]
    fig.add_annotation(
        x=max_entry["month"],
        y=max_entry["hours_played"],
        text=f"Peak: {max_entry['hours_played']:.1f} hours",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
    )

    return fig

# Weekly Listening Trends by Year: Patterns throughout the year
def create_weekly_trends_plot(weekly_listening):
# Create the figure with consistent styling
    fig = px.line(
        weekly_listening[1:-1],  # Still trimming incomplete edge weeks
        x="week",
        y="hours_played",
        color="year",
        title="<b>Weekly Listening Trends by Year</b><br><sup>Patterns throughout the year</sup>",
        labels={"hours_played": "Listening Hours", "week": "Week of Year"},
        color_discrete_sequence=color_palette,
        line_shape="spline",  # Smoother curves
        render_mode="svg",  # Sharper lines
    )

    # Update traces for better visualization
    fig.update_traces(
        line=dict(width=2.5),
        hovertemplate="<b>%{fullData.name}</b><br>Week: %{x}<br>Hours: %{y:.1f}<extra></extra>",
        mode="lines",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        xaxis=dict(
            title="<b>Week of Year</b>",
            tickmode="array",
            tickvals=list(range(1, 53, 4)),  # Show every 4th week
        ),
        yaxis=dict(
            title="<b>Listening Hours</b>", gridcolor="lightgrey", gridwidth=0.5
        ),
        legend=dict(
            title="Year:",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    # Add reference elements
    avg_hours = weekly_listening["hours_played"].mean()
    fig.add_hline(
        y=avg_hours,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Overall Avg: {avg_hours:.1f} hours",
        annotation_position="bottom right",
    )

    # Add seasonal annotations
    fig.add_vrect(
        x0=51,
        x1=53,
        fillcolor="blue",
        opacity=0.1,
        annotation_text="Winter",
        annotation_position="top left",
    )
    # Add seasonal annotations
    fig.add_vrect(
        x0=1,
        x1=12,
        fillcolor="blue",
        opacity=0.1,
        annotation_text="Winter",
        annotation_position="top left",
    )
    fig.add_vrect(
        x0=25,
        x1=38,
        fillcolor="red",
        opacity=0.1,
        annotation_text="Summer",
        annotation_position="top right",
    )

    return fig

# Average Daily Listening by Day of the Week
def create_dow_listening_plot(daily_listening, day_order):
 # Prepare data
    day_of_the_week_listening = (
        daily_listening.groupby("day_of_week")["hours_played"]
        .mean()
        .reindex(day_order)
        .reset_index()
    )
    day_of_the_week_listening.columns = ["day_of_week", "avg_hours"]

    # Create consistent color gradient (lighter to darker from Monday to Sunday)
    colors = px.colors.sequential.Blues[-len(day_order) :][
        ::-1
    ]  # Reverse to make Monday lightest

    fig = px.bar(
        day_of_the_week_listening,
        x="day_of_week",
        y="avg_hours",
        color="day_of_week",
        color_discrete_sequence=colors,
        title="<b>Average Daily Listening by Day of the Week</b>",
        labels={"avg_hours": "Average Hours", "day_of_week": "Day of Week"},
        category_orders={"day_of_week": day_order},
    )

    # Update traces
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f} hours<extra></extra>",
        # marker_line,
        width=0.5,
        marker_line_color="white",
        opacity=0.9,
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        showlegend=False,
        xaxis={
            "title": "<b>Day of Week</b>",
            "type": "category",
            "categoryorder": "array",
            "categoryarray": day_order,
        },
        yaxis={
            "title": "<b>Ave,rage Listening Hours</b>",
            "gridcolor": "lightgrey",
            "gridwidth": 0.5,
        },
        hovermode="x unified",
    )

    # Add average line
    avg_hours = day_of_the_week_listening["avg_hours"].mean()
    fig.add_hline(
        y=avg_hours,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Overall Avg: {avg_hours:.2f} hours",
        annotation_position="bottom right",
    )

    # Highlight peak day
    max_day = day_of_the_week_listening.loc[
        day_of_the_week_listening["avg_hours"].idxmax()
    ]
    fig.add_annotation(
        x=max_day["day_of_week"],
        y=max_day["avg_hours"],
        text=f"Peak: {max_day['avg_hours']:.2f} hours",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
    )

    return fig

# Average Listening Time by Hour of Day: Normalized by total number of days
def create_hourly_listening_plot(df):
# Convert seconds to minutes
    divMin = lambda x: x / 60

    # Calculate total listening time per hour
    plays_per_hour = df.groupby("hour")["secondsPlayed"].sum().apply(divMin)

    # Normalize by number of days
    num_days = df["date"].nunique()
    plays_per_hour = plays_per_hour / num_days

    # Prepare data for plotting
    hourly_data = plays_per_hour.reset_index()
    hourly_data.columns = ["hour", "avg_minutes"]

    # Create color gradient (cool to warm)
    colors = px.colors.sequential.Teal_r + px.colors.sequential.Oranges[1:]

    fig = px.bar(
        hourly_data,
        x="hour",
        y="avg_minutes",
        title="<b>Average Listening Time by Hour of Day</b><br><sup>Normalized by total number of days</sup>",
        labels={"avg_minutes": "Listening Time (min)", "hour": "Hour of Day"},
        color="avg_minutes",
        color_continuous_scale=colors,
    )

    # Update traces
    fig.update_traces(
        hovertemplate="<b>%{x}:00</b><br>Avg: %{y:.1f} min<extra></extra>",
        marker_line_width=0.5,
        marker_line_color="white",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        xaxis={
            "title": "<b>Hour of Day</b>",
            "tickvals": list(range(24)),
            "tickmode": "array",
        },
        yaxis={
            "title": "<b>Average Listening Time (min)</b>",
            "gridcolor": "lightgrey",
            "gridwidth": 0.5,
        },
        coloraxis_showscale=False,  # Hide color scale since hours are ordered
        bargap=0.05,
    )

    # Add average line
    avg_minutes = hourly_data["avg_minutes"].mean()
    fig.add_hline(
        y=avg_minutes,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Daily Avg: {avg_minutes:.1f} min",
        annotation_position="bottom right",
    )

    # Highlight peak hour
    max_hour = hourly_data.loc[hourly_data["avg_minutes"].idxmax()]
    fig.add_annotation(
        x=max_hour["hour"],
        y=max_hour["avg_minutes"],
        text=f"Peak: {max_hour['avg_minutes']:.1f} min",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
    )

    return fig

# Weekly Listening Patterns: Average minutes per hour by day of week
def create_hourly_heatmap(df, day_order):
   # Convert seconds to minutes
    divMin = lambda x: x / 60

    # Prepare data
    hourly_by_day = (
        df.groupby(["day_of_week", "hour"])["secondsPlayed"]
        .sum()
        .apply(divMin)
        .reset_index()
    )

    # Normalize by number of days (assuming equal days per week)
    n_days = df["date"].nunique()
    hourly_by_day["avg_minutes"] = hourly_by_day["secondsPlayed"] / (n_days / 7)

    # Create pivot table for heatmap
    heatmap_data = hourly_by_day.pivot(
        index="day_of_week", columns="hour", values="avg_minutes"
    ).reindex(day_order)

    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        origin="lower",
        title="<b>Weekly Listening Patterns</b><br><sup>Average minutes per hour by day of week</sup>",
        labels=dict(x="Hour of Day", y="Day of Week", color="Minutes"),
    )

    # Update traces
    fig.update_traces(
        hovertemplate="<b>%{y}, %{x}:00</b><br>Avg: %{z:.1f} min<extra></extra>",
        hoverongaps=False,
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        xaxis=dict(
            title="<b>Hour of Day</b>", tickvals=list(range(24)), tickmode="array"
        ),
        yaxis=dict(title="<b>Day of Week</b>", type="category"),
        coloraxis_colorbar=dict(title="Minutes", ticks="outside"),
        height=600,
    )

    return fig

# Weekday vs Weekend Listening Patterns: Average minutes per hour
def create_weekday_weekend_comparison(hourly_by_day):
 # Create weekday/weekend flag
    hourly_by_day["is_weekend"] = hourly_by_day["day_of_week"].isin(
        ["Saturday", "Sunday"]
    )

    # Aggregate data
    weekday_weekend = (
        hourly_by_day.groupby(["is_weekend", "hour"])["avg_minutes"]
        .mean()
        .reset_index()
    )

    # Create custom names for legend
    weekday_weekend["time_type"] = weekday_weekend["is_weekend"].map(
        {True: "Weekend", False: "Weekday"}
    )

    # Create plot
    fig = px.line(
        weekday_weekend,
        x="hour",
        y="avg_minutes",
        color="time_type",
        line_dash="time_type",
        title="<b>Weekday vs Weekend Listening Patterns</b><br><sup>Average minutes per hour</sup>",
        labels={"avg_minutes": "Average Minutes", "hour": "Hour of Day"},
        color_discrete_sequence=[color_palette[0], color_palette[5]],  # Distinct colors
    )

    # Update traces
    fig.update_traces(
        line=dict(width=2.5),
        hovertemplate="<b>%{fullData.name}</b><br>Hour: %{x}:00<br>Minutes: %{y:.1f}<extra></extra>",
        mode="lines",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        xaxis=dict(
            title="<b>Hour of Day</b>", tickvals=list(range(24)), tickmode="array"
        ),
        yaxis=dict(
            title="<b>Average Minutes Played</b>", gridcolor="lightgrey", gridwidth=0.5
        ),
        legend=dict(
            title="Time Type:",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    # Add peak annotations
    for is_weekend, color in [(False, color_palette[0]), (True, color_palette[5])]:
        subset = weekday_weekend[weekday_weekend["is_weekend"] == is_weekend]
        max_row = subset.loc[subset["avg_minutes"].idxmax()]
        fig.add_annotation(
            x=max_row["hour"],
            y=max_row["avg_minutes"],
            text=f"Peak: {max_row['avg_minutes']:.1f} min",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40 if is_weekend else 40,
            font=dict(color=color),
        )

    return fig

# Top {n} Nighttime Artists ({start_h}-{end_h} AM)
def create_nighttime_artists_plot(df, start_h=0, end_h=5, n=10):
# Filter nighttime listens
    night_listens = df[df["hour"].between(start_h, end_h)]

    # Get top artists
    top_night_artists = (
        night_listens["artistName"].value_counts().nlargest(n).reset_index()
    )
    top_night_artists.columns = ["artist", "count"]

    # Create plot
    fig = px.bar(
        top_night_artists,
        x="count",
        y="artist",
        title=f"<b>Top {n} Nighttime Artists ({start_h}-{end_h} AM)</b>",
        labels={"count": "Number of Plays", "artist": "Artist"},
        color_continuous_scale=color_palette,
        orientation="h",
    )

    # Update traces
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Plays: %{x}<extra></extra>",
        marker_line_width=0.5,
        marker_line_color="white",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="<b>Number of Plays</b>",
        yaxis_title="<b>Artist</b>",
        height=500,
        # margin=dict(l=120),  # Extra margin for artist names
    )

    return fig

# Top {n} Nighttime Tracks ({start_h}-{end_h} AM)
def create_nighttime_tracks_plot(df, start_h=0, end_h=5, n=10):
 # Filter nighttime listens
    night_listens = df[df["hour"].between(start_h, end_h)]

    # Get top tracks
    top_night_tracks = (
        night_listens["trackName"].value_counts().nlargest(n).reset_index()
    )
    top_night_tracks.columns = ["track", "count"]

    # Create plot
    fig = px.bar(
        top_night_tracks,
        x="count",
        y="track",
        title=f"<b>Top {n} Nighttime Tracks ({start_h}-{end_h} AM)</b>",
        labels={"count": "Number of Plays", "track": "Track Name"},
        # color="count",
        color_continuous_scale=color_palette,
        orientation="h",
    )

    # Update traces
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Plays: %{x}<extra></extra>",
        marker_line_width=0.5,
        marker_line_color="white",
    )

    # Update layout
    fig.update_layout(
        **common_layout,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="<b>Number of Plays</b>",
        yaxis_title="<b>Track Name</b>",
        height=500,
        # margin=dict(l=150),  # Extra margin for track names
    )

    return fig

# Nighttime Listening Activity ({start_h}-{end_h} AM): Daily patterns with highlighted peak nights
def create_nighttime_activity_plot(night_listens, start_h=0, end_h=5):
   # Prepare data - count tracks per night
    night_counts = (
        night_listens.resample("D", on="endTime")
        .size()
        .reset_index(name="tracks_played")
    )
    night_counts["date_str"] = night_counts["endTime"].dt.strftime("%b %d, %Y")

    # Create base plot with consistent styling
    fig = px.scatter(
        night_counts,
        x="endTime",
        y="tracks_played",
        opacity=0.7,
        hover_data=["date_str"],
        title=f"<b>Nighttime Listening Activity ({start_h}-{end_h} AM)</b><br><sup>Daily patterns with highlighted peak nights</sup>",
        labels={"endTime": "Date", "tracks_played": "Tracks Played"},
        color_discrete_sequence=[color_palette[3]],  # Using your established palette
    )

    # Add connecting line
    fig.add_trace(
        go.Scatter(
            x=night_counts["endTime"],
            y=night_counts["tracks_played"],
            mode="lines",
            opacity=0.7,
            line=dict(
                color=color_palette[3],
                width=2,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Highlight top nights (top 5% or max 7, whichever is smaller)
    n_top = min(7, max(1, int(len(night_counts) * 0.05)))
    top_nights = night_counts.nlargest(n_top, "tracks_played")

    fig.add_trace(
        go.Scatter(
            x=top_nights["endTime"],
            y=top_nights["tracks_played"],
            mode="markers",
            marker=dict(
                color=color_palette[0],  # Contrast color from your palette
                size=12,
                line=dict(width=2, color="white"),
            ),
            name="Peak Nights",
            hovertemplate="<b>%{text}</b><br>%{y} tracks<extra></extra>",
            text=top_nights["date_str"],
        )
    )

    # Update layout to match your style
    fig.update_layout(
        **common_layout,
        # plot_bgcolor="white",
        xaxis=dict(
            title="<b>Date</b>",
            gridcolor="lightgrey",
            showline=True,
            linecolor="lightgrey",
        ),
        yaxis=dict(
            title="<b>Tracks Played</b>",
            gridcolor="lightgrey",
            showline=True,
            linecolor="lightgrey",
        ),
        showlegend=False,
    )

    # Add smart annotations (avoid overlapping)
    y_range = night_counts["tracks_played"].max() - night_counts["tracks_played"].min()
    for i, row in top_nights.iterrows():
        fig.add_annotation(
            x=row["endTime"],
            y=row["tracks_played"],
            text=f"{row['tracks_played']} tracks",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=(-40 if i % 2 else 40),  # Alternate direction
            font=dict(color=color_palette[0], size=12),
            bgcolor="rgba(255,255,255,0.8)",
        )

    # Add reference line for average
    avg_tracks = night_counts["tracks_played"].mean()
    fig.add_hline(
        y=avg_tracks,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Average: {avg_tracks:.1f} tracks/night",
        annotation_position="bottom right",
    )

    return fig
