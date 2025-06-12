import os
from pathlib import Path
from yarl import URL
import openai
from dotenv import find_dotenv, load_dotenv

env_file = find_dotenv()
load_dotenv(env_file, override=True)

GIT_DATA_SET = "https://raw.githubusercontent.com/AleksNeStu/ai-real-estate-assistant/refs/heads/main/dataset/pl"
GIT_DATA_SET_URLS_STR = '\n'.join([GIT_DATA_SET + f'/apartments_rent_pl_2024_0{i}.csv' for i in range(1, 7)])