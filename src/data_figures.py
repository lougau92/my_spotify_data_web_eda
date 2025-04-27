import pandas as pd
import numpy as np
from datetime import datetime
import plotly
import plotly.express as px
import src.all_figures as all_figures


def split_songs_podcasts(df):
    """Split data into songs and podcasts based on duration threshold."""
    songs_thresh_min = 15
    df_songs = df.where(df["secondsPlayed"] < songs_thresh_min * 60.0).dropna()
    df_podcast = df.where(df["secondsPlayed"] >= songs_thresh_min * 60.0).dropna()
    return df_songs, df_podcast


def process_raw_data(data):
    """Process raw input data into cleaned DataFrame with additional features."""
    df = pd.DataFrame(data)

    # Convert and extract datetime features
    df["endTime"] = pd.to_datetime(df["endTime"])
    df["secondsPlayed"] = df["msPlayed"] / 1000

    # Extract various time features
    df["date"] = df["endTime"].dt.date
    df["hour"] = df["endTime"].dt.hour
    df["day_of_week"] = df["endTime"].dt.day_name()
    df["day_num"] = df["endTime"].dt.dayofweek  # Monday=0, Sunday=6
    df["month_name"] = df["endTime"].dt.month_name()
    df["month_year"] = df["endTime"].dt.to_period("M").astype(str)
    df["year"] = df["endTime"].dt.year

    # Clean up
    df.drop(columns=["msPlayed"], inplace=True)
    tot_skipped = len(df.where(df["secondsPlayed"] < 10))
    df.where(df["secondsPlayed"] > 10, inplace=True)

    return df.dropna(), tot_skipped


def create_play_count_figures(df_songs):
    """Create figures related to play counts and song popularity."""
    figures = []

    # Group by trackName and count the number of plays
    song_play_counts = df_songs["trackName"].value_counts().reset_index()
    song_play_counts.columns = ["trackName", "play_count"]
    figures.append(all_figures.create_play_count_distribution(song_play_counts))

    # Calculate and print play statistics
    play_count_distribution = song_play_counts["play_count"].value_counts().sort_index()
    print(
        "Total songs played more than once:",
        len(df_songs["trackName"].unique()),
        ",",
        (play_count_distribution.iloc[1:].sum() - play_count_distribution.iloc[1])
        / play_count_distribution.iloc[1:].sum(),
        "% of total records",
    )

    # Calculate replay probabilities
    total_songs = song_play_counts.shape[0]
    songs_played_more_than_once = song_play_counts[
        song_play_counts["play_count"] > 1
    ].shape[0]
    songs_played_more_than_twice = song_play_counts[
        song_play_counts["play_count"] > 2
    ].shape[0]
    songs_played_more_than_three_times = song_play_counts[
        song_play_counts["play_count"] > 3
    ].shape[0]

    # Print probabilities
    print(
        f"Probability of replay given played once: {songs_played_more_than_once / total_songs:.2%}"
    )
    print(
        f"Probability of replay given played twice: {songs_played_more_than_twice / songs_played_more_than_once:.2%}"
    )
    print(
        f"Probability of replay given played three times: {songs_played_more_than_three_times / songs_played_more_than_twice:.2%}"
    )

    return figures


def create_artist_figures(df_songs, n=15):
    """Create figures related to artist listening patterns."""
    figures = []

    # Get top artists
    top_artists = df_songs["artistName"].value_counts().nlargest(n)
    top_artists_fig, artist_colors = all_figures.create_top_artists_bar(
        df_songs, top_artists, n
    )
    figures.append(top_artists_fig)

    # Monthly artist plays
    artist_monthly = (
        df_songs.groupby(["month_year", "artistName"]).size().reset_index(name="count")
    )
    figures.append(all_figures.create_top_artist_monthly(artist_monthly, artist_colors))

    # Artist topic river
    df_top_artists = df_songs[df_songs["artistName"].isin(top_artists.index[:10])]
    df_artist_plays = (
        df_top_artists.groupby(["date", "artistName"]).size().unstack(fill_value=0)
    )
    df_artist_plays = df_artist_plays[
        df_artist_plays.sum().sort_values(ascending=False).index
    ]
    df_artist_plays.index = pd.to_datetime(df_artist_plays.index)
    df_artist_plays = df_artist_plays.resample("W").sum()
    figures.append(
        all_figures.create_artist_topic_river(df_artist_plays, artist_colors)
    )

    return figures


def create_track_figures(df_songs, n=15):
    """Create figures related to track listening patterns."""
    figures = []

    # Get top tracks
    top_tracks = df_songs["trackName"].value_counts().nlargest(n)
    top_tracks_fig, track_colors = all_figures.create_top_track_bar(
        df_songs, top_tracks, n
    )
    figures.append(top_tracks_fig)

    # Monthly track plays
    track_monthly = (
        df_songs.groupby(["month_year", "trackName"]).size().reset_index(name="count")
    )
    figures.append(all_figures.create_top_track_monthly(track_monthly, track_colors))

    # Track topic river
    df_top_tracks = df_songs[df_songs["trackName"].isin(top_tracks.index[:10])]
    df_tracks_plays = (
        df_top_tracks.groupby(["date", "trackName"]).size().unstack(fill_value=0)
    )
    df_tracks_plays = df_tracks_plays[
        df_tracks_plays.sum().sort_values(ascending=False).index
    ]
    df_tracks_plays.index = pd.to_datetime(df_tracks_plays.index)
    df_tracks_plays = df_tracks_plays.resample("W").sum()
    figures.append(all_figures.create_artist_topic_river(df_tracks_plays, track_colors))

    return figures


def create_aggregated_datasets(df):
    """Create various aggregated datasets for visualization."""

    # Convert seconds to hours
    def divHour(num):
        return num / 3600.0

    # Daily listening
    daily_listening = (
        df.groupby("date")["secondsPlayed"].sum().apply(divHour).reset_index()
    )
    daily_listening.columns = ["date", "hours_played"]
    daily_listening["date"] = pd.to_datetime(daily_listening["date"])
    daily_listening["year"] = daily_listening["date"].dt.year
    daily_listening["month"] = daily_listening["date"].dt.month_name()
    daily_listening["week"] = daily_listening["date"].dt.isocalendar().week
    daily_listening["day_of_week"] = daily_listening["date"].dt.day_name()

    # Add rolling averages
    for window, std in [(7, 3), (15, 3), (30, 5)]:
        daily_listening[f"{window}_day_avg"] = (
            daily_listening["hours_played"]
            .rolling(window, center=True, win_type="gaussian")
            .mean(std=std)
        )

    # Weekly, monthly, yearly aggregations
    weekly_listening = (
        daily_listening.groupby(["year", "week"])["hours_played"].sum().reset_index()
    )
    monthly_listening = (
        daily_listening.groupby(["year", "month"])["hours_played"].sum().reset_index()
    )
    yearly_listening = (
        daily_listening.groupby("year")["hours_played"].sum().reset_index()
    )

    return {
        "daily": daily_listening,
        "weekly": weekly_listening,
        "monthly": monthly_listening,
        "yearly": yearly_listening,
    }


def print_summary_statistics(aggregated_data):
    """Print summary statistics about listening habits."""
    daily_listening = aggregated_data["daily"]

    # Average listening time
    avg_listening_time = daily_listening["hours_played"].mean()
    print("Average Listening Time per Day: {:.2f} hours".format(avg_listening_time))

    # Top 5 days
    top_5_days = daily_listening.sort_values(by="hours_played", ascending=False).head(5)
    print("\nTop 5 Days with the Most Playing Time:")
    print(top_5_days[["date", "hours_played"]])

    # Sort by the 7-day rolling sum and get the top 5 windows TODO find non-overlaping weeks
    top_7_day_windows = daily_listening.sort_values(
        by="7_day_avg", ascending=False
    ).head(5)
    print("\nTop 7-Day Time Windows with the Most Playing Time:")
    print(top_7_day_windows)


def create_time_analysis_figures(df, aggregated_data):
    """Create figures related to time-based analysis."""
    figures = []
    daily_listening = aggregated_data["daily"]
    weekly_listening = aggregated_data["weekly"]
    monthly_listening = aggregated_data["monthly"]
    yearly_listening = aggregated_data["yearly"]

    # Define ordering
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Sort monthly data
    monthly_listening["month"] = pd.Categorical(
        monthly_listening["month"], categories=month_order, ordered=True
    )
    monthly_listening.sort_values(["year", "month"], inplace=True)

    # Create time-based figures
    figures.append(all_figures.create_yearly_listening_pie(yearly_listening))
    figures.append(all_figures.create_daily_listening_trends(daily_listening))
    figures.append(
        all_figures.create_monthly_trends_plot(monthly_listening, month_order)
    )
    figures.append(all_figures.create_weekly_trends_plot(weekly_listening[1:-1]))
    figures.append(all_figures.create_dow_listening_plot(daily_listening, day_order))
    figures.append(all_figures.create_hourly_listening_plot(df))
    figures.append(all_figures.create_hourly_heatmap(df, day_order))

    return figures


def create_special_analysis_figures(df, start_hour, end_hour):
    """Create figures for special analyses (weekday/weekend, nighttime)."""
    figures = []
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Hourly by day analysis
    hourly_by_day = (
        df.groupby(["day_of_week", "hour"])["secondsPlayed"]
        .sum()
        .apply(lambda x: x / 60.0)  # Convert to minutes
        .reset_index()
    )
    hourly_by_day["day_of_week"] = pd.Categorical(
        hourly_by_day["day_of_week"], categories=day_order, ordered=True
    )
    n_days = df["date"].nunique()
    hourly_by_day["avg_minutes"] = hourly_by_day["secondsPlayed"] / (n_days / 7)
    hourly_by_day["is_weekend"] = hourly_by_day["day_of_week"].isin(
        ["Saturday", "Sunday"]
    )

    # Create special analysis figures
    figures.append(all_figures.create_weekday_weekend_comparison(hourly_by_day))
    figures.append(all_figures.create_nighttime_artists_plot(df, start_hour, end_hour))
    figures.append(all_figures.create_nighttime_tracks_plot(df, start_hour, end_hour))

    # Nighttime analysis
    night_listens = df[df["hour"].between(start_hour, end_hour)]

    figures.append(all_figures.create_nighttime_activity_plot(night_listens))

    return figures


def build_figures(data):

    # Initialize DataFrame and process data
    df, tot_skipped = process_raw_data(data)

    print(
        "Total skipped tracks (<10s):",
        tot_skipped,
        ",",
        tot_skipped / (len(df)),
        "% of total records",
    )

    figures = []

    # Split data into songs and podcasts
    df_songs, df_podcast = split_songs_podcasts(df)

    figures.extend(create_play_count_figures(df_songs))
    figures.append(all_figures.create_duration_distribution(df_songs))

    # Number of unique artists and tracks
    print(f"Unique Artists: {df_songs['artistName'].nunique()}")
    print(f"Unique Tracks: {df['trackName'].nunique()}")

    figures.extend(create_artist_figures(df_songs))
    figures.extend(create_track_figures(df_songs))

    # Create aggregated datasets
    aggregated_data = create_aggregated_datasets(df)

    # Print summary statistics
    print_summary_statistics(aggregated_data)

    figures.extend(create_time_analysis_figures(df, aggregated_data))
    figures.extend(create_special_analysis_figures(df))

    # Convert figures to HTML divs
    return [plotly.offline.plot(fig, output_type="div") for fig in figures]
