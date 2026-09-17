from statistics import mean


SAMPLE_TARGET = 5


def next_sample_position(observation_count, sample_target=SAMPLE_TARGET):
    """Return (period_number, sample_number) for the next uploaded image."""
    return (
        (observation_count // sample_target) + 1,
        (observation_count % sample_target) + 1,
    )


def build_period_summaries(observations, sample_target=SAMPLE_TARGET):
    """Group ordered ClassificationLog rows and calculate period averages."""
    # Chunk by saved chronological order so existing records created by the
    # old "one image = one week" logic remain usable without a DB migration.
    grouped = {}
    for index, observation in enumerate(observations):
        period_number = (index // sample_target) + 1
        grouped.setdefault(period_number, []).append(observation)

    summaries = []
    previous_completed_average = None

    for period_number in sorted(grouped):
        period_observations = grouped[period_number]
        valid_lengths = [
            item.estimated_length_cm
            for item in period_observations
            if item.estimated_length_cm is not None
        ]

        completed = len(period_observations) >= sample_target
        average_length = (
            round(mean(valid_lengths), 2)
            if completed and valid_lengths
            else None
        )
        change_cm = None
        status = "COLLECTING SAMPLES"

        if completed and average_length is None:
            status = "MEASUREMENT UNAVAILABLE"
        elif completed and previous_completed_average is None:
            status = "BASELINE COMPLETE"
        elif completed:
            change_cm = round(
                average_length - previous_completed_average,
                2,
            )
            if change_cm > 0:
                status = "BATCH AVERAGE INCREASED"
            elif change_cm == 0:
                status = "NO AVERAGE SIZE CHANGE"
            else:
                status = "BATCH AVERAGE DECREASED"

        summaries.append({
            "period_number": period_number,
            "sample_count": len(period_observations),
            "sample_target": sample_target,
            "completed": completed,
            "average_length_cm": average_length,
            "change_cm": change_cm,
            "status": status,
            "observations": period_observations,
        })

        if completed and average_length is not None:
            previous_completed_average = average_length

    return summaries


def latest_period_summary(observations, sample_target=SAMPLE_TARGET):
    summaries = build_period_summaries(observations, sample_target)
    return summaries[-1] if summaries else None