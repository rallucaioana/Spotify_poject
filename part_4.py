import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

con = sqlite3.connect("data/spotify_database.db")
cur = con.cursor()