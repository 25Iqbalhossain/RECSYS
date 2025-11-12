# data/samples/make_history_dummy_domains.py
from pathlib import Path
import json, random

# ---- OUTPUT PATH ----
OUT = Path(r"C:\Users\hi\recsys\data\raw_events\user_history_external.jsonl")

# ---- CONFIG ----
NUM_USERS = 200
AVG_LEN = 25                # avg history length per user
RECENT_FIRST = True         # most-recent-first order
SEED = 7
random.seed(SEED)

# ---- DOMAIN VOCAB (তোমার দরকারমতো বাড়াও/কমাও) ----
DOMAINS = [
    "google.com","youtube.com","facebook.com","instagram.com","twitter.com","x.com",
    "tiktok.com","reddit.com","linkedin.com","whatsapp.com","pinterest.com",
    "netflix.com","amazon.com","twitch.tv","discord.com","snapchat.com",
    "github.com","stackOverflow.com","medium.com","quora.com",
    "bbc.com","cnn.com","nytimes.com","theverge.com","techcrunch.com",
    "udemy.com","coursera.org","khanacademy.org","wikipedia.org",
    "aliexpress.com","ebay.com","etsy.com","shopify.com","stripe.com",
    "spotify.com","soundcloud.com","apple.com","microsoft.com",
    "imdb.com","rottentomatoes.com","today.com","yahoo.com","bing.com",
]

PATH_FRAGMENTS = [
    "", "home", "feed", "watch", "trending", "explore", "news",
    "profile", "messages", "inbox", "learn/python", "learn/ml",
    "shop/deals", "topic/ai", "topic/sports", "topic/music",
]

def sample_url():
    d = random.choice(DOMAINS)
    frag = random.choice(PATH_FRAGMENTS)
    if frag and random.random() < 0.8:
        
        slug = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(4,10)))
        return f"{d}/{frag}/{slug}"
    return d

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for u in range(NUM_USERS):
            uid = f"user_{u:05d}"
            length = max(5, int(random.expovariate(1/AVG_LEN)) + 5)
            hist = [sample_url() for _ in range(length)]
        
            if RECENT_FIRST:
                hist = hist[::-1]  # most-recent-first

            rec = {"user_id": uid, "history": [str(x) for x in hist]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
