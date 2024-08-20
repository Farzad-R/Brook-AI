import os
import shutil
import sqlite3
import pandas as pd
import requests
from pyprojroot import here
from load_config import LoadConfig
CFG = LoadConfig()


def download_data(overwrite=False):
    """
    Downloads a SQLite database file from a remote URL, updates its datetime values, and saves it locally.

    This function performs the following steps:
    1. Checks if the SQLite database file (`local_file`) and its backup (`backup_file`) already exist. If they do not exist or if the `overwrite` flag is set to `True`, it downloads the database from the specified URL and creates a backup.
    2. Reads the downloaded database (`local_file`) into pandas DataFrames for manipulation.
    3. Adjusts datetime fields in the `flights` and `bookings` tables to reflect the current time.
    4. Writes the modified DataFrames back to the `local_file`, replacing the existing tables.

    The `backup_file` serves as a backup to restore the database to its original state if needed but is not modified during the process. 

    Parameters:
        overwrite (bool): If `True`, the function will download the database file and create a backup even if the files already exist. If `False` (default), it will skip the download and backup if the files are already present.

    Raises:
        requests.HTTPError: If the request to download the database fails.
        sqlite3.DatabaseError: If there is an issue with reading or writing the SQLite database.
        pandas.errors.ParserError: If there is an issue with parsing datetime fields in the DataFrames.
    """
    db_url = CFG.travel_db_url
    # db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    local_file = here("data/travel2.sqlite")
    # The backup lets us restart for each tutorial section
    backup_file = here("data/travel2.backup.sqlite")
    if overwrite or not os.path.exists(local_file):
        print("downloading the data...")
        response = requests.get(db_url)
        response.raise_for_status()  # Ensure the request was successful
        with open(local_file, "wb") as f:
            f.write(response.content)
        # Backup - we will use this to "reset" our DB in each section
        shutil.copy(local_file, backup_file)
    else:
        print("Data alreay exists on the directory.")
    # Convert the flights to present time for our tutorial
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace(
            "\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace(
                "\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()


# db = local_file  # We'll be using this local file as our DB in this tutorial
if __name__ == "__main__":
    download_data()
