# ascii_visualizer.py

def ascii_train_timeline(train_schedule, trains):
    """
    Prints a simple ASCII timeline of train movements.
    Time in minutes is scaled horizontally.
    """
    time_scale = 2  # 1 character = 2 minutes
    max_time = 0
    for schedule in train_schedule.values():
        for arrival, _ in schedule.values():
            if arrival > max_time:
                max_time = arrival

    print("\nTrain Movement Timeline (ASCII)\n")
    for train in trains:
        train_id = train["id"]
        line = f"{train_id:5}: "
        timeline = [" "] * (max_time // time_scale + 10)
        prev_time = train["departure"]
        prev_station = train["start"]

        schedule = train_schedule.get(train_id, {})
        for station, (arrival, platform) in schedule.items():
            start_idx = prev_time // time_scale
            end_idx = arrival // time_scale
            for i in range(start_idx, end_idx):
                timeline[i] = "#"
            # Mark station arrival with platform
            timeline[end_idx] = str(platform)
            prev_time = arrival
            prev_station = station

        line += "".join(timeline)
        print(line)

    # Time scale guide
    print("\nLegend: # = train moving, number = platform at station arrival\n")
