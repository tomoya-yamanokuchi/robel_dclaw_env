from datetime import datetime


def time_as_string():
    dt_now = datetime.now()
    dt_str = "{}{}{}{}{}{}".format(
        dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute, dt_now.second
    )
    return dt_str
