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
    df = df.dropna()
    tot = len(df)
    df = df[df["secondsPlayed"] > 10]
    tot_skipped = tot - len(df)

    return df, tot_skipped


def create_play_count_figures(df_songs):
    """Create figures related to play counts and song popularity."""
    figures = []

    # Group by trackName and count the number of plays
    song_play_counts = df_songs["trackName"].value_counts().reset_index()
    song_play_counts.columns = ["trackName", "play_count"]
    figures.append(all_figures.create_play_count_distribution(song_play_counts))

    # Calculate and print play statistics
    play_count_distribution = song_play_counts["play_count"].value_counts().sort_index()
    more_then_once = (
        play_count_distribution.iloc[1:].sum() - play_count_distribution.iloc[1]
    ) / play_count_distribution.iloc[1:].sum()

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

    p1 = songs_played_more_than_once / total_songs
    p2 = songs_played_more_than_twice / songs_played_more_than_once
    p3 = songs_played_more_than_three_times / songs_played_more_than_twice

    return figures, p1, p2, p3


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

    # Top 5 days
    top_5_days = daily_listening.sort_values(by="hours_played", ascending=False).head(5)

    # Sort by the 7-day rolling sum and get the top 5 windows TODO find non-overlaping weeks
    top_7_day_windows = daily_listening.sort_values(
        by="7_day_avg", ascending=False
    ).head(1)

    # rename 7_day_avg col by

    return (
        avg_listening_time,
        top_5_days[["date", "hours_played"]],
        top_7_day_windows[["date", "7_day_avg"]],
    )


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
    listening_trend = all_figures.compute_listening_trend(daily_listening)
    figures.append(
        all_figures.create_monthly_trends_plot(monthly_listening, month_order)
    )
    figures.append(all_figures.create_weekly_trends_plot(weekly_listening[1:-1]))
    figures.append(all_figures.create_dow_listening_plot(daily_listening, day_order))
    figures.append(all_figures.create_hourly_listening_plot(df))
    figures.append(all_figures.create_hourly_heatmap(df, day_order))

    return figures, listening_trend


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

    # Split data into songs and podcasts
    df_songs, df_podcast = split_songs_podcasts(df)

    play_count_figs, p1, p2, p3 = create_play_count_figures(df_songs)
    duration_dist_figs = all_figures.create_duration_distribution(df_songs)

    artist_figs = create_artist_figures(df_songs)
    track_figs = create_track_figures(df_songs)

    # Create aggregated datasets
    aggregated_data = create_aggregated_datasets(df)

    # Print summary statistics
    avg_listening_time, top_5_days, top_7_day_windows = print_summary_statistics(
        aggregated_data
    )

    time_figs, listening_trend = create_time_analysis_figures(df, aggregated_data)
    special_figs = create_special_analysis_figures(df, 0, 5)

    html_figures = []

    # Section Header
    html_figures.append(
        """
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #2a3f5f; font-family: Arial; margin-bottom: 10px;'>
            1. Understanding Your Listening Habits
        </h1>
        <p style='text-align: center; color: #555; font-size: 1.1em; max-width: 800px; margin: 0 auto;'>
            We begin by exploring the big picture of your Spotify journey - how your listening patterns 
            have evolved over time and what they reveal about your music consumption.
        </p>
    </div>
    """
    )

    # Yearly Distribution
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>1.1</span> Annual Listening Patterns
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='max-width: 1500px; margin: 0 auto;'>
                {yearly_fig}
            </div>
        </div>
    </div>
    """.format(
            yearly_fig=plotly.offline.plot(time_figs[0], output_type="div"),
        )
    )

    trend = ""
    if listening_trend == 1:
        trend = "gradual increase in listening time"
    elif listening_trend == 0:
        trend = "stable"
    else:
        trend = "gradual decrease in listening time"

    # Daily Trends with Scrollable Container
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>1.2</span> Daily Listening Rhythms
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px; text-align: center;'>
            <div style='display: inline-block; width: 100%; max-width: 1500px; overflow-x: auto;'>
                <div style='min-width: 1000px; margin: 0 auto;'>
                    {daily_fig}
                </div>
            </div>
            <div style='display: flex; justify-content: center; margin: 20px auto 0; max-width: 1500px;'>
                <div style='width: 48%; padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin: 0 10px;'>
                    <p style='color: #555; margin: 0;'>
                        <strong>Last year trend:</strong> {trend_description}
                    </p>
                </div>
            </div>
        </div>
    </div>
    """.format(
            daily_fig=plotly.offline.plot(
                time_figs[1],
                output_type="div",
                config={"responsive": False},
            ),
            trend_description=trend,
        )
    )

    # Section 2 Header
    html_figures.append(
        """
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #2a3f5f; font-family: Arial; margin-bottom: 10px;'>
            2. Your Musical Preferences
        </h1>
        <p style='text-align: center; color: #555; font-size: 1.1em; max-width: 800px; margin: 0 auto;'>
            Explore what captures your ears - from all-time favorites to evolving monthly trends
        </p>
    </div>
    """
    )

    # Top Artists & Tracks - Side by Side
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>2.1</span> All-Time Favorites
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;'>
                <div style='flex: 1; min-width: 400px; max-width: 750px;'>
                    {top_artists_fig}
                    <div style='margin-top: 15px; padding: 12px; background-color: #f8f9fa; border-radius: 5px;'>
                        <p style='color: #555; margin: 0; text-align: center;'>
                            <strong>Top Artist:</strong> {top_artist} ({top_artist_count} plays)
                        </p>
                    </div>
                </div>
                <div style='flex: 1; min-width: 400px; max-width: 750px;'>
                    {top_tracks_fig}
                    <div style='margin-top: 15px; padding: 12px; background-color: #f8f9fa; border-radius: 5px;'>
                        <p style='color: #555; margin: 0; text-align: center;'>
                            <strong>Top Track:</strong> {top_track} ({top_track_count} plays)
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """.format(
            top_artists_fig=plotly.offline.plot(artist_figs[0], output_type="div"),
            top_tracks_fig=plotly.offline.plot(track_figs[0], output_type="div"),
            top_artist=artist_figs[0].data[0].y[-1] if len(artist_figs) > 0 else "N/A",
            top_artist_count=(
                int(artist_figs[0].data[0].x[-1]) if len(artist_figs) > 0 else "N/A"
            ),
            top_track=track_figs[0].data[0].y[-1] if len(track_figs) > 0 else "N/A",
            top_track_count=(
                int(track_figs[0].data[0].x[-1]) if len(track_figs) > 0 else "N/A"
            ),
        )
    )

    # Monthly Breakdown
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>2.2</span> Monthly Favorites
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;'>
                <div style='flex: 1; min-width: 1000px; max-width:1500px;'>
                    {monthly_artists_fig}
                </div>
                <div style='flex: 1; min-width:1000px; max-width: 1500px;'>
                    {monthly_tracks_fig}
                </div>
            </div>
        </div>
    </div>
    """.format(
            monthly_artists_fig=plotly.offline.plot(artist_figs[1], output_type="div"),
            monthly_tracks_fig=plotly.offline.plot(track_figs[1], output_type="div"),
        )
    )

    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>2.3</span> Listening Evolution
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px; text-align: center;'>
            <!-- Header -->
            <div style='text-align: center; margin-bottom: 30px;'>
                <p style='color: #555; margin: 0; font-size: 1.1em;'>
                    How your favorite artists and tracks have changed over time
                </p>
            </div>
            
            <div style='margin-bottom: 40px; text-align: center;'>
                <div style='width: 100%; display: flex; justify-content: center;'>
                    <div style='width: 95%; max-width: 1500px; overflow: hidden;'>
                        <div style='margin: 0 auto; width: 100%;'>
                            {topic_river_artists_fig}
                        </div>
                    </div>
                </div>
            </div>
            
            <div style='margin-top: 40px; text-align: center;'>
                <div style='width: 100%; display: flex; justify-content: center;'>
                    <div style='width: 95%; max-width: 1500px; overflow: hidden;'>
                        <div style='margin: 0 auto; width: 100%;'>
                            {topic_river_tracks_fig}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    """.format(
            topic_river_artists_fig=plotly.offline.plot(
                artist_figs[2],
                output_type="div",
                config={"responsive": False},
            ),
            topic_river_tracks_fig=plotly.offline.plot(
                track_figs[2],
                output_type="div",
                config={"responsive": False},
            ),
        )
    )

    # Section 3 Header
    html_figures.append(
        """
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #2a3f5f; font-family: Arial; margin-bottom: 10px;'>
            3. Your Listening Rhythms
        </h1>
        <p style='text-align: center; color: #555; font-size: 1.1em; max-width: 800px; margin: 0 auto;'>
            Discover the patterns in when you listen - from seasonal changes to your daily routines
        </p>
    </div>
    """
    )

    # Monthly Trends
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>3.1</span> Seasonal Listening Patterns - Monthly
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='max-width: 1500px; margin: 0 auto;'>
                {month_trends_fig}
            </div>
            <div style='max-width: 800px; margin: 20px auto 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
                <p style='color: #555; margin: 0; text-align: center;'>
                    <strong>Seasonal Insight:</strong> Your listening peaks in {peak_month} with {peak_value}% above average
                </p>
            </div>
        </div>
    </div>
    """.format(
            month_trends_fig=plotly.offline.plot(
                time_figs[2].update_layout(height=500), output_type="div"
            ),
            peak_month=(
                time_figs[2].data[0].x[np.argmax(time_figs[2].data[0].y)]
                if len(time_figs) > 2
                else "N/A"
            ),
            peak_value=(
                int(
                    ((time_figs[2].data[0].y.max() / time_figs[2].data[0].y.mean()) - 1)
                    * 100
                )
                if len(time_figs) > 2
                else "N/A"
            ),
        )
    )

    # Weekly Trends
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>3.1</span> Seasonal Listening Patterns - Weekly
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='max-width: 1500px; margin: 0 auto;'>
                {month_trends_fig}
            </div>
            <div style='max-width: 800px; margin: 20px auto 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
                <p style='color: #555; margin: 0; text-align: center;'>
                    <strong>Seasonal Insight:</strong> Your listening peaks in {peak_month} with {peak_value}% above average
                </p>
            </div>
        </div>
    </div>
    """.format(
            month_trends_fig=plotly.offline.plot(
                time_figs[3].update_layout(height=500), output_type="div"
            ),
            peak_month=(
                time_figs[3].data[0].x[np.argmax(time_figs[3].data[0].y)]
                if len(time_figs) > 3
                else "N/A"
            ),
            peak_value=(
                int(
                    ((time_figs[3].data[0].y.max() / time_figs[3].data[0].y.mean()) - 1)
                    * 100
                )
                if len(time_figs) > 3
                else "N/A"
            ),
        )
    )

    # total by week day
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>3.1</span> Daily Listenings
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='max-width: 1500px; margin: 0 auto;'>
                {month_trends_fig}
            </div>
        </div>
    </div>
    """.format(
            month_trends_fig=plotly.offline.plot(
                time_figs[4].update_layout(height=500), output_type="div"
            ),
        )
    )

    # Hourly Patterns (Dual Visualization)
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>3.3</span> Daily Listening Pulse
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='justify-content: center; gap: 30px; margin-bottom: 20px;'>
                <div style='flex: 1; min-width: 400px;'>
                    <h3 style='color: #444; font-family: Arial; text-align: center;'>Average Hourly Patterns</h3>
                    {hourly_listening_fig}
                </div>
                <div style='flex: 1; min-width: 400px;'>
                    <h3 style='color: #444; font-family: Arial; text-align: center;'>Weekday vs Weekend</h3>
                    {weekday_weekend_fig}
                </div>
            </div>
            <div style='max-width: 800px; margin: 0 auto; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
                <p style='color: #555; margin: 0; text-align: center;'>
                    <strong>Key Difference:</strong> You listen to {weekend_diff}% more music on weekends between {peak_diff_hours}
                </p>
            </div>
        </div>
    </div>
    """.format(
            weekday_weekend_fig=plotly.offline.plot(special_figs[0], output_type="div"),
            hourly_listening_fig=plotly.offline.plot(time_figs[5], output_type="div"),
            weekend_diff=abs(
                (
                    int(
                        (
                            (
                                special_figs[0].data[1].y.mean()
                                / special_figs[0].data[0].y.mean()
                            )
                            - 1
                        )
                        * 100
                    )
                    if len(special_figs) > 0
                    else "N/A"
                ),
            ),
            peak_diff_hours=(
                f"{np.argmax(special_figs[0].data[1].y - special_figs[0].data[0].y)}:00-{np.argmax(special_figs[0].data[1].y - special_figs[0].data[0].y)+1}:00"
                if len(special_figs) > 0
                else "N/A"
            ),
        ),
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    hm_data = time_figs[6].data[0].z.copy()
    hm_data[np.isnan(hm_data)] = 0

    # Weekly Heatmap (Full Width)
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>3.2</span> Weekly Listening Energy
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='width: 100%; overflow-x: auto;'>
            <div style='max-width: 1500px; margin: 0 auto;'>
                    {weekly_heatmap_fig}
                </div>
            </div>
            <div style='max-width: 800px; margin: 20px auto 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
                <p style='color: #555; margin: 0; text-align: center;'>
                    <strong>Pattern:</strong> Your most active listening occurs on {peak_day} around {peak_hour}
                </p>
            </div>
        </div>
    </div>
    """.format(
            weekly_heatmap_fig=plotly.offline.plot(
                time_figs[6].update_layout(height=500),
                output_type="div",
                # config={"responsive": False},
            ),
            peak_day=(
                day_order[np.unravel_index(np.argmax(hm_data), hm_data.shape)[0]]
                if len(time_figs) > 6
                else "N/A"
            ),
            peak_hour=(np.argmax(hm_data) if len(time_figs) > 6 else "N/A"),
        )
    )

    # Section 4 Nighttime Listening
    html_figures.append(
        """
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #2a3f5f; font-family: Arial; margin-bottom: 10px;'>
            4. Nighttime Listening (0-5AM)
        </h1>
        <p style='text-align: center; color: #555; font-size: 1.1em; max-width: 800px; margin: 0 auto;'>
            A Special focus on your late-night habits
        </p>
    </div>
    """
    )

    # Nighttime Listening Section
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>4.1</span> Activity Timeline 
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='max-width: 1500px; margin: 0 auto 30px;'>
                {nighttime_activity_fig}
            </div>
        </div>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>4.1</span> Listening Content 
            </h2>            
            <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 30px; margin-top: 30px;'>
                <div min-width: 400px; max-width: 750px;'>
                    {nighttime_artists_fig}
                </div>
                <div min-width: 400px; max-width: 750px;'>
                    {nighttime_tracks_fig}
                </div>
            </div>
        </div>
    </div>
    """.format(
            nighttime_activity_fig=plotly.offline.plot(
                special_figs[3], output_type="div"
            ),
            nighttime_artists_fig=plotly.offline.plot(
                special_figs[1], output_type="div"
            ),
            nighttime_tracks_fig=plotly.offline.plot(
                special_figs[2], output_type="div"
            ),
        )
    )

    # Section 5 Deep Dives
    html_figures.append(
        """
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #2a3f5f; font-family: Arial; margin-bottom: 10px;'>
            5. Key Metrics
        </h1>
        <p style='text-align: center; color: #555; font-size: 1.1em; max-width: 800px; margin: 0 auto;'>
            a comprehensive listening statistics
        </p>
    </div>
    """
    )

    # Extract the `data from the figure
    data = duration_dist_figs.data[0].x

    # Get bin settings (if custom bins are used)
    if (
        hasattr(duration_dist_figs.data[0], "xbins")
        and duration_dist_figs.data[0].xbins is not None
    ):
        bin_size = duration_dist_figs.data[0].xbins.size
        min_val, max_val = np.min(data), np.max(data)
        bins = np.arange(min_val, max_val + bin_size, bin_size)
    else:
        bins = "auto"  # Let numpy choose bins automatically

    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=bins)

    # Find the most frequent bin
    max_count_idx = np.argmax(counts)
    max_count = counts[max_count_idx]
    most_frequent_bin = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2

    # Calculate percentage
    total_counts = np.sum(counts)
    percentage = (max_count / total_counts) * 100

    # Technical Insights Section
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>5.1</span> Your Listening By The Numbers
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 30px;'>
                <div style='flex: 1; min-width: 250px; padding: 15px; background-color: #e6f7ff; border-radius: 8px;'>
                    <h3 style='color: #2a3f5f; font-family: Arial; text-align: center;'>Average Session</h3>
                    <p style='font-size: 1.8em; color: #636efa; text-align: center; margin: 10px 0;'>
                        {avg_listening_time:.1f} hours
                    </p>
                    <p style='color: #666; text-align: center; margin: 0;'>
                        per day
                    </p>
                </div>
                
                <div style='flex: 1; min-width: 250px; padding: 15px; background-color: #fff2e6; border-radius: 8px;'>
                    <h3 style='color: #2a3f5f; font-family: Arial; text-align: center;'>Music Diversity</h3>
                    <p style='font-size: 1.8em; color: #ff7f0e; text-align: center; margin: 10px 0;'>
                        {artists:,} unique artists
                    </p>
                    <p style='font-size: 1.8em; color: #ff7f0e; text-align: center; margin: 10px 0;'>
                        {tracks:,} unique tracks
                    </p>
                </div>
                
                <div style='flex: 1; min-width: 250px; padding: 15px; background-color: #e6ffed; border-radius: 8px;'>
                    <h3 style='color: #2a3f5f; font-family: Arial; text-align: center;'>Content Mix</h3>
                    <p style='font-size: 1.8em; color: #2ca02c; text-align: center; margin: 10px 0;'>
                        {songs:,} songs
                    </p>
                    <p style='font-size: 1.8em; color: #2ca02c; text-align: center; margin: 10px 0;'>
                        {podcasts:,} podcasts
                    </p>
                </div>
            </div>
            
            <div style='max-width: 1500px; margin: 30px auto 0;'>
                {duration_dist_fig}
                <div style='max-width: 600px; margin: 20px auto 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
                    <p style='color: #555; margin: 0; text-align: center;'>
                        <strong>Typical Session:</strong> Most songs ({peak_duration_pct:.0f}%) are {peak_duration:2f} minutes long
                    </p>
                </div>
            </div>
        </div>
    </div>
    """.format(
            avg_listening_time=avg_listening_time,
            artists=df_songs["artistName"].nunique(),
            tracks=df["trackName"].nunique(),
            songs=len(df_songs),
            podcasts=len(df_podcast),
            duration_dist_fig=plotly.offline.plot(
                duration_dist_figs, output_type="div"
            ),
            peak_duration_pct=percentage,
            peak_duration=most_frequent_bin,
        )
    )

    # Song Play Patterns
    html_figures.append(
        """
    <div style='margin-bottom: 40px;'>
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>5.2</span> Your Music Replay Behavior
            </h2>
        </div>
        <div style='border: 1px solid #f0f2f6; padding: 20px; border-radius: 0 0 8px 8px;'>
            <div style='max-width: 1500px; margin: 0 auto;'>
                {play_count_fig}
            </div>
            <div style='max-width: 1500px; margin: 20px auto 0;'>
                <div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>
                    <div style='width: 100%; margin-bottom: 15px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
                        <p style='color: #555; margin: 0; text-align: center;'>
                            <strong>Your Listening Personality:</strong> {listening_type}
                        </p>
                    </div>
                    <div style='width: 32%; padding: 12px; background-color: #e6f7ff; border-radius: 5px;'>
                        <p style='color: #555; margin: 0; text-align: center;'>
                            <strong>Replayed at least Once</strong><br>{p1:.0f}% chance
                        </p>
                    </div>
                    <div style='width: 32%; padding: 12px; background-color: #e6f7ff; border-radius: 5px;'>
                        <p style='color: #555; margin: 0; text-align: center;'>
                            <strong>Replayed Once → Twice</strong><br>{p2:.0f}% chance
                        </p>
                    </div>
                    <div style='width: 32%; padding: 12px; background-color: #e6f7ff; border-radius: 5px;'>
                        <p style='color: #555; margin: 0; text-align: center;'>
                            <strong>Replayed Twice → Thrice</strong><br>{p3:.0f}% chance
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """.format(
            play_count_fig=plotly.offline.plot(play_count_figs[0], output_type="div"),
            p1=p1 * 100,
            p2=p2 * 100,
            p3=p3 * 100,
            listening_type=(
                "Explorer (you prefer discovering new music)"
                if p1 < 0.3
                else "Repeater (you enjoy revisiting favorites)"
            ),
        )
    )

    html_figures.append(
        f"<h2 style='text-align: center; color: #555;'>From all listened tracks, you skipped a total {tot_skipped} (<10s), which represents {tot_skipped / (len(df) *100):.2%} of the total records.</h2>"
    )

    # Peak Listening Days - Enhanced Version
    html_figures.append(
        """
    <div style='flex: 1; min-width: 400px;'>
    
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px 8px 0 0;'>
            <h2 style='color: #2a3f5f; font-family: Arial; margin: 0;'>
                <span style='color: #636efa;'>5.3</span> Your Top Listening Periods
            </h2>
        </div>   
        <div style='background-color: #636efa; padding: 15px; color: white; font-family: Arial;'>
            <h3 style='margin: 0; text-align: center; font-size: 1.1em;'>
                <i class='fas fa-trophy' style='margin-right: 8px;'></i>
                Your Top 5 Listening Days
            </h3>
        </div>
        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
            <table style='width: 100%; border-collapse: collapse; font-family: Arial;'>
                <thead>
                    <tr  style='color: #636efa;'>
                        <th style='padding: 12px; text-align: left;'>Date</th>
                        <th style='padding: 12px; text-align: right;'>Hours Played</th>
                        <th style='padding: 12px; text-align: center;'>Day of Week</th>
                    </tr>
                </thead>
                <tbody>
                    {top_days_rows}
                </tbody>
            </table>
    
        </div>
    </div>
    """.format(
            top_days_rows="\n".join(
                [
                    f"""
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px;'>{row['date'].strftime('%b %d, %Y')}</td>
            <td style='padding: 10px; text-align: right; font-weight: bold;'>{row['hours_played']:.1f} hours</td>
            <td style='padding: 10px; text-align: center; color: #666;'>{row['date'].strftime('%A')}</td>
        </tr>
        """
                    for _, row in top_5_days.iterrows()
                ]
            ),
        )
    )

    # Record-Breaking Week - Centered Layout
    html_figures.append(
        """
    <div style='flex: 1; min-width: 400px; margin: 0 15px;'>
        <div style='background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); overflow: hidden;'>
            <div style='background-color: #ff7f0e; padding: 15px; color: white; font-family: Arial;'>
                <h3 style='margin: 0; text-align: center; font-size: 1.1em;'>
                    <i class='fas fa-trophy' style='margin-right: 8px;'></i>
                    Your Record-Breaking Week
                </h3>
            </div>
            <div style='padding: 20px;'>
                <div style='text-align: center; margin-bottom: 20px;'>
                    <div style='font-size: 2.2em; font-weight: bold; color: #ff7f0e; line-height: 1;'>
                        {peak_week_avg:.1f}
                    </div>
                    <div style='color: #666; margin-top: 5px;'>hours/day average</div>
                    <div style='color: #888; font-size: 0.9em; margin-top: 3px;'>
                        {peak_week_start} - {peak_week_end}
                    </div>
                </div>
                
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;'>
                    <div style='background-color: #f8f9fa; padding: 12px; border-radius: 5px; text-align: center;'>
                        <div style='font-size: 0.9em; color: #666;'>Compared to average</div>
                        <div style='font-weight: bold; color: {diff_color}; margin-top: 3px;'>
                            {diff_percent:.0f}% {diff_direction}
                        </div>
                    </div>
                    <div style='background-color: #f8f9fa; padding: 12px; border-radius: 5px; text-align: center;'>
                        <div style='font-size: 0.9em; color: #666;'>Total week hours</div>
                        <div style='font-weight: bold; color: #333; margin-top: 3px;'>
                            {total_hours:.0f}
                        </div>
                    </div>
                </div>
                
                <div style='padding: 12px; background-color: #f8f9fa; border-radius: 5px; text-align: center;'>
                    <p style='color: #555; margin: 0; font-size: 0.9em;'>
                        <i class='fas fa-calendar-check' style='color: #ff7f0e; margin-right: 5px;'></i>
                        {week_context}
                    </p>
                </div>
            </div>
        </div>
    </div>
    """.format(
            peak_week_avg=top_7_day_windows.iloc[0]["7_day_avg"],
            peak_week_start=top_7_day_windows.iloc[0]["date"].strftime("%b %d"),
            peak_week_end=(
                top_7_day_windows.iloc[0]["date"] + pd.Timedelta(days=6)
            ).strftime("%b %d"),
            diff_percent=abs(
                (top_7_day_windows.iloc[0]["7_day_avg"] / avg_listening_time - 1) * 100
            ),
            diff_direction=(
                "higher"
                if top_7_day_windows.iloc[0]["7_day_avg"] > avg_listening_time
                else "lower"
            ),
            diff_color=(
                "#2ca02c"
                if top_7_day_windows.iloc[0]["7_day_avg"] > avg_listening_time
                else "#d62728"
            ),
            total_hours=top_7_day_windows.iloc[0]["7_day_avg"] * 7,
            week_context=(
                "Your most active listening week during this period"
                if not top_7_day_windows.empty
                else ""
            ),
        )
    )

    #     Track/Artist Evolution

    #         Show how favorites changed over time:
    #         "Your 2022 favorites vs 2023 favorites"
    #         (create_top_artist_monthly & create_top_track_monthly)

    #     Session Length Analysis

    #         Include if you have the session length plot from earlier discussions

    # Convert figures to HTML divs
    return html_figures
