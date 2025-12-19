import pandas as pd
import ast
import re

# ---- Step 1: Load raw dataset ----
df = pd.read_csv("may_july_chunk_1.csv")
df2 = pd.read_csv("may_july_chunk_15.csv")
df3 = pd.read_csv("may_july_chunk_287.csv")
df4 = pd.read_csv("may_july_chunk_392.csv")

df = pd.concat([df, df2, df3, df4], ignore_index=True)

# ---- Step 2: Extract user_id from 'user' column ----
def extract_user_id(user_str):
    try:
        # Remove datetime objects from the string
        clean_str = re.sub(r"datetime\.datetime\([^)]*\)", "'DATE'", user_str)
        user_dict = ast.literal_eval(clean_str)
        return user_dict.get("id_str", None)
    except Exception:
        return None

df["user_id"] = df["user"].apply(extract_user_id)

# ---- Step 3: Extract mentioned user IDs ----
def extract_mentions(mention_str):
    try:
        mentions_list = ast.literal_eval(mention_str)
        if isinstance(mentions_list, list):
            return [m.get("id_str", None) for m in mentions_list if isinstance(m, dict)]
        return []
    except Exception:
        return []

df["mention_ids"] = df["mentionedUsers"].apply(extract_mentions)

# ---- Step 4: Convert epoch → proper datetime ----
df["created_at"] = pd.to_datetime(df["epoch"], unit="s")

# ---- Step 5: Flatten viewCount ----
def parse_viewcount(vc_str):
    try:
        vc_dict = ast.literal_eval(vc_str)
        return int(vc_dict.get("count", 0))
    except Exception:
        return 0

if "viewCount" in df.columns:
    df["viewCount"] = df["viewCount"].apply(parse_viewcount)

# ---- Step 6: Select useful columns ----
keep_cols = [
    "user_id", "text", "hashtags", "mention_ids",
    "in_reply_to_user_id_str", "created_at",
    "likeCount", "retweetCount", "replyCount"
]

if "viewCount" in df.columns:
    keep_cols.append("viewCount")

filtered_df = df[keep_cols]

# ---- Step 7: Save cleaned dataset ----
filtered_df.to_csv("filtered_may_july.csv", index=False)

# ---- Step 8: Preview ----
print(filtered_df.head(10))
print("\nFinal dataset saved as 'filtered_may_july.csv' and is graph-ready!")
# Look at the raw "user" column to see actual structure
print(df["user"].head(5).tolist())
