#!/usr/bin/python3

# Calculates summary statistics over experiment results.
#
# usage: summarize.py [--all <dirname> | --list <files...>] 
# --all   produce summaries for all experiments in the dir
# --list  produce summary over the provided list of files

import pandas as pd
import argparse
import sys
import os

rep_ending = "-rep_000.txt"

flag = sys.argv[1]

#arg_parser = argparse.ArgumentParser(prog="summarize.py", help="Calculate summary statistics from optimization replications.")
#arg_parser.add_argument("--all", "-a", action="store_true", default=False)
#arg_parser.add_argument("--list", "-l", action="store_true", default=False)

# buckets = np.arange(0, max_time, 1)
# start by printing max time for each repliaction

def is_summary_file(name):
  return name.endswith("_mean.txt") or name.endswith("_var.txt")\
    or name.endswith("_var_time.txt") or name.endswith("_mean_time.txt")\
    or name.endswith("_median.txt") or name.endswith("_median_time.txt")\
    or name.endswith("_final_params.txt")

def max_rep_from_files(files: list) -> int:
  return 1 + max(
    map(int, [
        file[len(file) - len("000.txt"):len(file)-len(".txt")]
        for file in files
      ]) 
  )

def clean_cols(df):
  return df.drop(df.filter(regex=r"x\d|deriv_norm").columns, axis=1)

# clean unused columns and ensure that df has a granularity of 
# bucket_size in cumulative_time by forward-filling the values
def prepare_time(df, bucket_size, min_end_ctime):
  df = clean_cols(df)
  # convert cumulative_time to datetime and set as index
  df.cumulative_time = pd.to_datetime(df.cumulative_time.astype(float) / 1e6, unit="s")
  df = df.set_index(df.cumulative_time)
  df = df.drop(columns="cumulative_time")
  # resample to a frequency of bucket_size
  df = df.resample(pd.Timedelta(seconds=bucket_size)).mean().ffill()
  df = df.reset_index()
  # convert back to microseconds
  df.cumulative_time = df.cumulative_time.astype(int) / 1e3
  #df = df[df.cumulative_time <= min_end_ctime]
  return df

def summarize_files(files: list, basename: str):
  for idx, file in enumerate(sorted(files)):
    if os.path.getsize(file) <= 0:
      print(f"Warning: no data for experiment (yet)! ({idx}) Skipping experiment.")
      return

  # collect all resampled datasets and summarize over time by grouping
  freq = 3 #s

  max_start_ctime = 0
  for fname in files:
    max_start_ctime = max(pd.read_csv(fname).cumulative_time.min(), max_start_ctime)

  min_end_ctime = float("inf")
  for fname in files:
    min_end_ctime = min(pd.read_csv(fname).cumulative_time.max(), min_end_ctime)

  min_step = float("inf")
  for fname in files:
    min_step = min(pd.read_csv(fname).step.max(), min_step)

  time = pd.concat([prepare_time(pd.read_csv(file), freq, min_end_ctime) for file in files])
  time = time[time.cumulative_time >= max_start_ctime]
  time = time[time.cumulative_time <= min_end_ctime]
  time = time.groupby(time.cumulative_time)
  time.mean().to_csv(basename + "_mean_time.txt")
  time.median().to_csv(basename + "_median_time.txt")
  time.var().to_csv(basename + "_var_time.txt")
  ## summarize time by bucketing
  #time.cumulative_time = pd.to_datetime(time.cumulative_time.astype(float), unit="s")
  #time_mean = time[["cumulative_time", "y"]].resample(pd.Timedelta(seconds=freq), on="cumulative_time").mean()
  #time_var = time[["cumulative_time", "y"]].resample(pd.Timedelta(seconds=freq), on="cumulative_time").var()
  ## mean
  #time_mean.index = time_mean.index.astype(int) / 1e9
  #time_mean.to_csv(basename + "_time_mean.txt")
  ## var
  #time_var.index = time_mean.index.astype(int) / 1e9
  #time_var.to_csv(basename + "_time_var.txt")

  # summarize steps by grouping
  all_data = pd.concat([clean_cols(pd.read_csv(file)) for file in files])
  all_data = all_data[all_data.step <= min_step]
  all_data.drop(columns="cumulative_time")
  all_data = all_data.groupby(all_data.step)
  if len(all_data) == 0:
    print("Warning: Dataframe empty! Skipping experiment.")
    return
  nreps = len(files)
  max_rep_nr = max_rep_from_files(files)
  if max_rep_nr != nreps:
    print("Warning: some replications seem to be missing. The maximum replication number != #files!")
  # mean
  mean_df = all_data.mean() 
  mean_df["nreps"] = nreps
  mean_df.to_csv(basename + "_mean.txt")

  # median
  median_df = all_data.median() 
  median_df["nreps"] = nreps
  median_df.to_csv(basename + "_median.txt")

  # variance
  var_df = all_data.var()
  var_df["nreps"] = nreps
  var_df.to_csv(basename + "_var.txt")

# produce summary of all experiments in a particular directory
if flag == "--all":
  folder = sys.argv[2]
  filenames = [
    file for file in list(
      map(lambda f: os.path.join(folder, f), os.listdir(folder)) # full path to all fules in folder
    )
    if not is_summary_file(file) # don't include summaries!
  ]
  experiment_names = list(set([file[:-len(rep_ending)] for file in filenames])) # get just the basename
  for experiment in experiment_names:
    print(f"summarizing experiment {experiment}")
    experiment_filenames = [file for file in filenames if file.startswith(experiment)]
    summarize_files(experiment_filenames, experiment)
# produce summary over the provided list of files
elif flag == "--list":
  files = sys.argv[2:]
  summarize_files(files, files[0][:-len(rep_ending)])

