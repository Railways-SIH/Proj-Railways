# visualize_schedule.py
import matplotlib.pyplot as plt

def plot_train_schedule(train_schedule, trains):
    """
    Visualize train movements on a timeline.
    X-axis: Time (minutes)
    Y-axis: Trains
    Blocks are represented as colored segments
    """
    fig, ax = plt.subplots(figsize=(10, len(trains)*1.5))

    y_labels = []
    y_pos = []
    color_map = ["skyblue", "lightgreen", "salmon", "orange", "violet"]

    for idx, train in enumerate(trains):
        train_id = train["id"]
        y_labels.append(train_id)
        y_pos.append(idx*10)  # spacing between trains

        schedule = train_schedule.get(train_id, {})
        prev_time = train["departure"]
        prev_station = train["start"]

        color_idx = 0
        for station, (arrival, platform) in schedule.items():
            # Draw a segment for this block
            ax.barh(
                y=y_pos[idx],
                width=arrival - prev_time,
                left=prev_time,
                height=5,
                color=color_map[color_idx % len(color_map)],
                edgecolor="black",
                label=f"Block {prev_station}->{station}" if idx==0 else ""
            )
            # Label platform
            ax.text(
                x=prev_time + (arrival - prev_time)/2,
                y=y_pos[idx]+1,
                s=f"P{platform}",
                ha="center",
                va="bottom",
                fontsize=8
            )

            prev_time = arrival
            prev_station = station
            color_idx += 1

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Train Movement Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
