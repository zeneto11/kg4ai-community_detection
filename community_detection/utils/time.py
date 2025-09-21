
def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
